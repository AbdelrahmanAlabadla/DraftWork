const AUTH_CONFIG_URL = "/api/v1/auth/config";
let clerkPromise = null;
let lastClaimedSessionId = null;

function loadScript(src, attributes = {}) {
  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = src;
    script.async = true;
    script.crossOrigin = "anonymous";
    Object.entries(attributes).forEach(([name, value]) => script.setAttribute(name, value));
    script.onload = resolve;
    script.onerror = () => reject(new Error(`Could not load ${src}`));
    document.head.appendChild(script);
  });
}

function frontendDomain(publishableKey) {
  const encoded = publishableKey.split("_")[2];
  if (!encoded) throw new Error("Invalid Clerk publishable key");
  return atob(encoded).slice(0, -1);
}

export async function initClerk() {
  if (clerkPromise) return clerkPromise;
  clerkPromise = (async () => {
    const response = await fetch(AUTH_CONFIG_URL, { credentials: "same-origin" });
    if (!response.ok) throw new Error("Could not load authentication configuration");
    const config = await response.json();
    if (!config.enabled || !config.publishable_key) return null;

    const domain = config.frontend_api_url || frontendDomain(config.publishable_key);
    const origin = domain.startsWith("http") ? domain : `https://${domain}`;
    await loadScript(`${origin}/npm/@clerk/ui@1/dist/ui.browser.js`);
    await loadScript(
      `${origin}/npm/@clerk/clerk-js@6/dist/clerk.browser.js`,
      { "data-clerk-publishable-key": config.publishable_key }
    );
    await window.Clerk.load({
      ui: { ClerkUI: window.__internal_ClerkUICtor },
      signInUrl: "/sign-in.html",
      signUpUrl: "/sign-up.html",
      afterSignOutUrl: "/",
    });
    return window.Clerk;
  })();
  return clerkPromise;
}

export async function authHeaders(headers = {}) {
  try {
    const clerk = await initClerk();
    const token = clerk?.session ? await clerk.session.getToken() : null;
    return token ? { ...headers, Authorization: `Bearer ${token}` } : { ...headers };
  } catch (_) {
    return { ...headers };
  }
}

export async function claimSession() {
  const clerk = await initClerk();
  if (!clerk?.isSignedIn || !clerk.session) return false;
  if (lastClaimedSessionId === clerk.session.id) return true;
  const response = await fetch("/api/v1/auth/claim", {
    method: "POST",
    headers: await authHeaders(),
    credentials: "same-origin",
  });
  if (!response.ok) throw new Error("Could not attach this DraftWork session to your account");
  lastClaimedSessionId = clerk.session.id;
  return true;
}

export async function signOut() {
  const clerk = await initClerk();
  if (!clerk?.isSignedIn) return;
  await fetch("/api/v1/auth/logout", {
    method: "POST",
    headers: await authHeaders(),
    credentials: "same-origin",
  });
  await clerk.signOut({ redirectUrl: "/" });
}

function renderNavigation(clerk) {
  const signedOut = document.getElementById("authSignedOut");
  const signedIn = document.getElementById("authSignedIn");
  if (!signedOut || !signedIn) return;
  signedOut.hidden = Boolean(clerk?.isSignedIn);
  signedIn.hidden = !clerk?.isSignedIn;
  const mount = document.getElementById("clerkUserButton");
  if (clerk?.isSignedIn && mount && !mount.childElementCount) {
    clerk.mountUserButton(mount, {
      afterSignOutUrl: "/",
      customMenuItems: [
        { label: "manageAccount" },
        {
          label: "My Exams",
          onClick: () => {
            document.dispatchEvent(new CustomEvent("draftwork:open-my-exams"));
          },
          mountIcon: (element) => { element.textContent = "📄"; },
          unmountIcon: (element) => { if (element) element.textContent = ""; },
        },
        { label: "signOut" },
      ],
    });
  }
}

export async function initAuthNavigation() {
  let clerk;
  try {
    clerk = await initClerk();
  } catch (_) {
    renderNavigation(null);
    return;
  }
  renderNavigation(clerk);
  if (!clerk) return;
  let wasSignedIn = Boolean(clerk.isSignedIn);
  if (clerk.isSignedIn) await claimSession().catch(() => {});
  clerk.addListener(async () => {
    renderNavigation(clerk);
    if (clerk.isSignedIn) await claimSession().catch(() => {});
    if (wasSignedIn && !clerk.isSignedIn) {
      // Any Clerk sign-out path, including UserButton, asks FastAPI to replace
      // the account-owned cookie with a clean anonymous DraftWork session.
      await fetch("/api/v1/documents", { credentials: "same-origin" }).catch(() => {});
    }
    wasSignedIn = Boolean(clerk.isSignedIn);
  });
}

import { initClerk } from "./auth.js";

const target = document.getElementById("clerkAuth");
const loading = document.getElementById("authLoading");
const requestedRedirect = new URLSearchParams(window.location.search).get("redirect_url");
const redirectUrl = (
  requestedRedirect?.startsWith("/") && !requestedRedirect.startsWith("//")
) ? requestedRedirect : "/";
const clerkAppearance = {
  variables: {
    colorPrimary: "#f36e15",
    colorBackground: "#1c1f22",
    colorInputBackground: "#23272b",
    colorInputText: "#f3f4f6",
    colorText: "#f3f4f6",
    colorTextSecondary: "#c0c4ca",
    borderRadius: "10px",
  },
  elements: {
    cardBox: { width: "100%", boxShadow: "none" },
    card: { width: "100%", boxShadow: "none", padding: "0" },
    headerTitle: { display: "none" },
    headerSubtitle: { display: "none" },
    formFieldLabel: { color: "#ffffff" },
    formFieldInput: {
      color: "#ffffff",
      caretColor: "#ffffff",
      borderColor: "#5f6670",
    },
    formFieldInputShowPasswordButton: { color: "#ffffff" },
    formFieldAction: {
      color: "#ff8a3d",
      textDecoration: "underline",
      textUnderlineOffset: "3px",
    },
    formFieldHintText: { color: "#ffffff" },
    identityPreviewText: { color: "#ffffff" },
    identityPreviewEditButton: { color: "#ffffff" },
    badge: {
      backgroundColor: "#f36e15",
      borderColor: "#f36e15",
      color: "#ffffff",
      fontWeight: "700",
    },
    dividerText: { color: "#ffffff" },
    socialButtonsBlockButton: {
      backgroundColor: "#ffffff",
      borderColor: "#ffffff",
      color: "#151719",
    },
    socialButtonsBlockButtonText: { color: "#151719", fontWeight: "600" },
    footer: { background: "transparent" },
    footerActionText: { color: "#d7d9dd" },
    footerActionLink: { color: "#ff8a3d", fontWeight: "700" },
  },
};
let clerk = null;
try {
  clerk = await initClerk();
} catch (_) {
  if (loading) loading.hidden = true;
  if (target) {
    target.classList.add("auth-error");
    target.textContent = "Account access is temporarily unavailable. You can continue anonymously and try again later.";
  }
}
if (!clerk || !target) {
  if (loading) loading.hidden = true;
  if (target && !target.textContent) target.textContent = "Authentication is not configured.";
} else if (clerk.isSignedIn) {
  window.location.replace(redirectUrl);
} else if (document.body.dataset.authPage === "signup") {
  if (loading) loading.hidden = true;
  clerk.mountSignUp(target, {
    signInUrl: "/sign-in.html",
    forceRedirectUrl: redirectUrl,
    appearance: clerkAppearance,
  });
} else {
  if (loading) loading.hidden = true;
  clerk.mountSignIn(target, {
    signUpUrl: "/sign-up.html",
    forceRedirectUrl: redirectUrl,
    appearance: clerkAppearance,
  });
}

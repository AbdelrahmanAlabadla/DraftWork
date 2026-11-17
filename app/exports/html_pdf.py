"""Production HTML-to-PDF export using the shared Exam Review renderer.

The browser receives canonical Document IR and renders it through the same
``exam-view.js`` functions and CSS used by the interactive preview.  Python
does not lay out, wrap, shape, or draw question content.
"""
from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app import config
from app.exports.document_ir import build_document_ir

if TYPE_CHECKING:
    from playwright.async_api import Browser, Playwright, Route


_FRONTEND_DIR = Path(__file__).resolve().parents[2] / "FrontEnd"
_PRINT_PAGE = _FRONTEND_DIR / "print.html"


class HtmlPdfError(RuntimeError):
    """Raised when Chromium cannot produce a verified PDF."""


async def _allow_local_assets_only(route: Route) -> None:
    if route.request.url.startswith("file:"):
        await route.continue_()
    else:
        await route.abort()


def _system_chromium() -> str | None:
    configured = config.PDF_CHROMIUM_EXECUTABLE
    if configured:
        return configured
    for name in ("chromium", "chromium-browser", "google-chrome", "chrome", "msedge"):
        resolved = shutil.which(name)
        if resolved:
            return resolved
    if config.APP_ENV != "production":
        candidates = (
            Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
            Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        )
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
    return None


class HtmlPdfRenderer:
    """Lazily starts one browser and reuses it across bounded export tasks."""

    def __init__(self) -> None:
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._start_lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(config.PDF_EXPORT_CONCURRENCY)

    def _bind_to_running_loop(self) -> None:
        loop = asyncio.get_running_loop()
        if self._loop is loop:
            return
        # A production ASGI worker keeps one loop for its lifetime.  TestClient
        # and some CLI harnesses create a fresh loop per request; Playwright
        # channel objects from a closed loop cannot be reused there.
        self._loop = loop
        self._playwright = None
        self._browser = None
        self._start_lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(config.PDF_EXPORT_CONCURRENCY)

    async def _ensure_browser(self) -> Browser:
        if self._browser is not None and self._browser.is_connected():
            return self._browser
        async with self._start_lock:
            if self._browser is not None and self._browser.is_connected():
                return self._browser
            try:
                if self._playwright is None:
                    from playwright.async_api import async_playwright

                    self._playwright = await async_playwright().start()
                executable = _system_chromium()
                launch_options: dict[str, Any] = {
                    "headless": True,
                    "args": [
                        "--allow-file-access-from-files",
                        "--disable-dev-shm-usage",
                        "--font-render-hinting=none",
                    ],
                }
                if executable:
                    launch_options["executable_path"] = executable
                self._browser = await self._playwright.chromium.launch(**launch_options)
            except Exception as exc:
                raise HtmlPdfError(
                    "Chromium is unavailable. Install the Playwright Chromium runtime."
                ) from exc
            return self._browser

    async def render_document(self, document_ir: dict[str, Any]) -> bytes:
        self._bind_to_running_loop()
        async with self._semaphore:
            browser = await self._ensure_browser()
            context = await browser.new_context(
                viewport={"width": 1440, "height": 1000},
                device_scale_factor=1,
            )
            await context.route("**/*", _allow_local_assets_only)
            page = await context.new_page()
            page.set_default_timeout(config.PDF_EXPORT_TIMEOUT_SECONDS * 1000)
            try:
                await page.goto(_PRINT_PAGE.as_uri(), wait_until="load")
                await page.wait_for_function(
                    "typeof window.renderDraftworkPrintDocument === 'function'"
                )
                metrics = await page.evaluate(
                    "async (documentIR) => {"
                    " await window.renderDraftworkPrintDocument(documentIR);"
                    " await document.fonts.ready;"
                    " const images = [...document.images];"
                    " await Promise.all(images.map((image) => image.complete"
                    "   ? Promise.resolve()"
                    "   : new Promise((resolve, reject) => {"
                    "       image.addEventListener('load', resolve, {once: true});"
                    "       image.addEventListener('error', reject, {once: true});"
                    "     })));"
                    " return window.getDraftworkPrintMetrics();"
                    "}",
                    document_ir,
                )
                expected = sum(
                    len(section.get("items") or [])
                    for section in document_ir.get("sections") or []
                )
                actual = int(metrics.get("items") or 0)
                if actual != expected:
                    raise HtmlPdfError(
                        f"Print DOM is incomplete: expected {expected} items, rendered {actual}."
                    )
                language = str(document_ir.get("language") or "en")
                if language == "ar":
                    footer_text = (
                        '<span dir="rtl">صفحة '
                        '<span class="pageNumber"></span> من '
                        '<span class="totalPages"></span></span>'
                    )
                else:
                    footer_text = (
                        'Page <span class="pageNumber"></span> of '
                        '<span class="totalPages"></span>'
                    )
                footer = (
                    '<div style="width:100%;font-family:Arial,sans-serif;'
                    'font-size:8px;color:#000;text-align:center">'
                    f"{footer_text}</div>"
                )
                payload = await page.pdf(
                    format="A4",
                    print_background=True,
                    prefer_css_page_size=True,
                    display_header_footer=True,
                    header_template="<span></span>",
                    footer_template=footer,
                    margin={"top": "10mm", "right": "0", "bottom": "16mm", "left": "0"},
                    tagged=True,
                )
                if not payload.startswith(b"%PDF"):
                    raise HtmlPdfError("Chromium returned an invalid PDF payload.")
                return payload
            except asyncio.TimeoutError as exc:
                raise HtmlPdfError("HTML-to-PDF rendering timed out.") from exc
            except HtmlPdfError:
                raise
            except Exception as exc:
                raise HtmlPdfError(f"HTML-to-PDF rendering failed: {exc}") from exc
            finally:
                await context.close()

    async def close(self) -> None:
        self._bind_to_running_loop()
        async with self._start_lock:
            browser, playwright = self._browser, self._playwright
            self._browser = None
            self._playwright = None
            try:
                if browser is not None:
                    await browser.close()
            finally:
                if playwright is not None:
                    await playwright.stop()


renderer = HtmlPdfRenderer()


async def render_exam_pdf(exam: dict[str, Any], metadata: dict[str, Any]) -> bytes:
    return await renderer.render_document(build_document_ir(exam, metadata, kind="student"))


async def render_answers_pdf(exam: dict[str, Any], metadata: dict[str, Any]) -> bytes:
    return await renderer.render_document(build_document_ir(exam, metadata, kind="answers"))


def render_exam_pdf_sync(exam: dict[str, Any], metadata: dict[str, Any]) -> bytes:
    """CLI/test helper; production requests use the reusable async renderer."""
    async def run() -> bytes:
        local_renderer = HtmlPdfRenderer()
        try:
            document = build_document_ir(exam, metadata, kind="student")
            return await local_renderer.render_document(document)
        finally:
            await local_renderer.close()
            # Let Windows Proactor subprocess pipe-close callbacks run before
            # asyncio.run() tears down this short-lived CLI/test loop.
            await asyncio.sleep(0.05)

    return asyncio.run(run())


def render_answers_pdf_sync(exam: dict[str, Any], metadata: dict[str, Any]) -> bytes:
    """CLI/test helper; production requests use the reusable async renderer."""
    async def run() -> bytes:
        local_renderer = HtmlPdfRenderer()
        try:
            document = build_document_ir(exam, metadata, kind="answers")
            return await local_renderer.render_document(document)
        finally:
            await local_renderer.close()
            await asyncio.sleep(0.05)

    return asyncio.run(run())


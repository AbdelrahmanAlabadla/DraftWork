from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class InvalidPdfError(ValueError):
    pass


def validate_pdf(path: str | Path) -> None:
    """Validate the complete PDF container without imposing a page-count limit."""
    candidate = Path(path)
    try:
        with candidate.open("rb") as source:
            if source.read(5) != b"%PDF-":
                raise InvalidPdfError("The file does not have a PDF signature")
        reader = PdfReader(str(candidate), strict=False)
        if reader.is_encrypted:
            try:
                unlocked = reader.decrypt("")
            except Exception as exc:
                raise InvalidPdfError("Encrypted PDFs are not supported") from exc
            if not unlocked:
                raise InvalidPdfError("Encrypted PDFs are not supported")
        if reader.trailer.get("/Root") is None:
            raise InvalidPdfError("The PDF document catalog is missing")
        # Resolve the page tree to detect broken cross-references. There is
        # intentionally no maximum page-count policy.
        len(reader.pages)
    except InvalidPdfError:
        raise
    except (PdfReadError, OSError, ValueError, TypeError) as exc:
        raise InvalidPdfError("The uploaded PDF could not be read") from exc

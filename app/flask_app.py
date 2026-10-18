"""Flask UI: PDF upload, problem input, cited report display, Markdown download.

Chalane ka tareeqa (project root se):
    python -m app.flask_app
phir browser mein http://127.0.0.1:5000 kholo.
"""
import os
import secrets
import threading
from datetime import datetime
from pathlib import Path

from flask import (
    Flask, flash, jsonify, redirect, render_template, request, send_from_directory, url_for,
)
from werkzeug.utils import secure_filename

from src.ingestion.indexer import get_vectorstore, index_pdf
from src.services.report import SAFETY_NOTES
from src.workflow.graph import run_workflow

MAX_QUERY_CHARS = 500


def friendly_error(exc: Exception) -> str:
    """Technical error ko saaf, mehfooz message mein badalta hai (stack trace user ko nahi dikhta)."""
    msg = str(exc)
    low = msg.lower()
    if isinstance(exc, RuntimeError) and "missing" in low:
        return "An API key is missing. Check GROQ_API_KEY and GOOGLE_API_KEY in your .env file."
    if "429" in msg or "resource_exhausted" in low or "rate_limit" in low or "rate limit" in low:
        return "The API rate limit was reached. Wait about a minute and try again."
    if "401" in msg or "403" in msg or "invalid_api_key" in low or "api key" in low or "permission" in low:
        return "An API key was rejected. Check the keys in your .env file."
    if "connect" in low or "timeout" in low or "timed out" in low:
        return "Could not reach the API. Check your internet connection and try again."
    if isinstance(exc, ValueError):
        return "This PDF has no extractable text (it may be a scanned image)."
    return "Something went wrong while processing the request. Details were written to the server log."


def list_documents() -> dict[str, int]:
    """Index mein maujood documents aur unke chunks ki tadaad."""
    data = get_vectorstore().get(include=["metadatas"])
    counts: dict[str, int] = {}
    for meta in data["metadatas"] or []:
        counts[meta["document"]] = counts.get(meta["document"], 0) + 1
    return dict(sorted(counts.items()))


def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=os.getenv("FLASK_SECRET_KEY") or secrets.token_hex(16),
        MAX_CONTENT_LENGTH=30 * 1024 * 1024,  # 30 MB se bari upload reject
        UPLOAD_DIR=Path(os.getenv("RAW_PDF_DIR", "data/raw_pdfs")),
        REPORT_DIR=Path("reports"),
    )
    if test_config:
        app.config.update(test_config)

    jobs: dict[str, dict] = {}  # background indexing ki halat
    lock = threading.Lock()

    def indexing_now() -> bool:
        return any(j["status"] == "indexing" for j in jobs.values())

    def safe_documents() -> tuple[dict[str, int], str | None]:
        try:
            return list_documents(), None
        except Exception as exc:
            app.logger.exception("Could not list documents")
            return {}, friendly_error(exc)

    def render_page(*, error=None, result=None, query="", status=200):
        docs, docs_error = safe_documents()
        return (
            render_template(
                "index.html",
                docs=docs,
                jobs=dict(jobs),
                indexing=indexing_now(),
                error=error or docs_error,
                result=result,
                query=query,
                max_chars=MAX_QUERY_CHARS,
            ),
            status,
        )

    @app.get("/")
    def index():
        return render_page()

    @app.get("/status")
    def status():
        with lock:
            return jsonify(jobs=dict(jobs))

    # ---------- upload + background indexing ----------
    def run_job(name: str, path: Path, force: bool) -> None:
        try:
            n = index_pdf(path, force=force)
            outcome = {"status": "done", "message": f"Indexed {n} chunks."}
        except Exception as exc:
            app.logger.exception("Indexing failed for %s", name)
            outcome = {"status": "error", "message": friendly_error(exc)}
        with lock:
            jobs[name] = outcome

    @app.post("/upload")
    def upload():
        file = request.files.get("pdf")
        if not file or not file.filename:
            flash("Choose a PDF file first.", "error")
            return redirect(url_for("index"))

        name = secure_filename(file.filename)
        if not name or not name.lower().endswith(".pdf"):
            flash("Only PDF files are supported.", "error")
            return redirect(url_for("index"))

        head = file.stream.read(5)
        file.stream.seek(0)
        if head != b"%PDF-":
            flash("This file is not a valid PDF.", "error")
            return redirect(url_for("index"))

        with lock:
            if indexing_now():
                flash("Another document is still being indexed. Wait until it finishes.", "error")
                return redirect(url_for("index"))
            upload_dir = Path(app.config["UPLOAD_DIR"])
            upload_dir.mkdir(parents=True, exist_ok=True)
            path = upload_dir / name
            file.save(path)

            docs, _ = safe_documents()
            # Pichhli koshish fail hui thi -> resume (force nahi); warna same naam = replace
            force = name in docs and jobs.get(name, {}).get("status") != "error"
            jobs[name] = {
                "status": "indexing",
                "message": "Indexing... this can take a few minutes on the free API tier.",
            }

        threading.Thread(target=run_job, args=(name, path, force), daemon=True).start()
        flash(f"{name} uploaded. Indexing started.", "ok")
        return redirect(url_for("index"))

    # ---------- diagnose ----------
    @app.post("/diagnose")
    def diagnose():
        query = " ".join(request.form.get("query", "").split())

        if not query:
            return render_page(error="Describe the equipment problem first.", status=400)
        if len(query) > MAX_QUERY_CHARS:
            return render_page(error=f"Keep the description under {MAX_QUERY_CHARS} characters.", query=query, status=400)
        if indexing_now():
            return render_page(error="A document is still being indexed. Wait until it finishes.", query=query, status=409)
        docs, _ = safe_documents()
        if not docs:
            return render_page(error="Upload and index at least one manual first.", query=query, status=400)

        try:
            state = run_workflow(query)
        except Exception as exc:
            app.logger.exception("Workflow failed")
            return render_page(error=friendly_error(exc), query=query, status=503)

        report_dir = Path(app.config["REPORT_DIR"])
        report_dir.mkdir(parents=True, exist_ok=True)
        filename = f"report_{datetime.now():%Y%m%d_%H%M%S}.md"
        (report_dir / filename).write_text(state["final_report"], encoding="utf-8")

        plan = state.get("diagnostic_plan")
        ver = state.get("verification_result")
        findings = state.get("findings", [])
        result = {
            "findings": findings,
            "steps": plan.steps if plan else [],
            "revision": plan.revision if plan else 0,
            "verification": ver,
            "unsupported": {c.id: c.reason for c in ver.unsupported()} if ver else {},
            "chunks": state.get("retrieved_chunks", []),
            "insufficient": not findings,
            "report_file": filename,
            "safety_notes": SAFETY_NOTES,
        }
        return render_page(result=result, query=query)

    @app.get("/reports/<path:filename>")
    def download_report(filename: str):
        # send_from_directory ../ jaisi traversal khud rok deta hai (404)
        return send_from_directory(Path(app.config["REPORT_DIR"]).resolve(), filename, as_attachment=True)

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000, debug=False)
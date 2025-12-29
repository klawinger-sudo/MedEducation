"""Gradio Web UI for MedEducation.

A clean, modern interface for querying medical textbooks with RAG.
"""

import os
from pathlib import Path
from typing import Generator, List, Tuple, Optional

import gradio as gr

from mededucation.chat.engine import ChatEngine
from mededucation.prompts.system import PROFILES


# Custom CSS for medical/professional theme
CUSTOM_CSS = """
.message-wrap {
    max-width: 100% !important;
}
.source-box {
    background: #f0f4f8;
    border-radius: 8px;
    padding: 12px;
    margin-top: 8px;
    font-size: 0.9em;
    border-left: 4px solid #2563eb;
}
.dark .source-box {
    background: #1e293b;
    border-left-color: #3b82f6;
}
.profile-badge {
    display: inline-block;
    padding: 4px 12px;
    background: #2563eb;
    color: white;
    border-radius: 16px;
    font-size: 0.85em;
    margin-bottom: 8px;
}
footer {
    display: none !important;
}
"""


class MedEducationUI:
    """Gradio-based web interface for MedEducation."""

    def __init__(
        self,
        vectordb_path: str = "./data/vectordb",
        config_path: str = "./config/sources.yaml",
        default_profile: str = "flight_critical_care",
    ):
        self.vectordb_path = vectordb_path
        self.config_path = config_path
        self.default_profile = default_profile
        self._engine: Optional[ChatEngine] = None
        self._current_profile = default_profile

    def _get_engine(self, profile: str = None) -> ChatEngine:
        """Get or create the chat engine."""
        profile = profile or self._current_profile

        if self._engine is None or self._current_profile != profile:
            self._current_profile = profile
            self._engine = ChatEngine(
                vectordb_path=self.vectordb_path,
                config_path=self.config_path,
                profile=profile,
            )
        return self._engine

    def get_sources(self) -> List[str]:
        """Get list of available sources."""
        try:
            engine = self._get_engine()
            sources = engine.get_sources()
            return ["All Sources"] + [s["source_id"] for s in sources]
        except Exception:
            return ["All Sources"]

    def get_profiles(self) -> List[str]:
        """Get list of available profiles."""
        return list(PROFILES.keys())

    def get_profile_info(self, profile: str) -> str:
        """Get profile description."""
        if profile in PROFILES:
            return f"**{PROFILES[profile]['name']}**\n\n{PROFILES[profile]['description']}"
        return ""

    def chat(
        self,
        message: str,
        history: List[Tuple[str, str]],
        source_filter: str,
        profile: str,
    ) -> Generator[Tuple[List[Tuple[str, str]], str], None, None]:
        """Process a chat message and stream the response."""
        if not message.strip():
            yield history, ""
            return

        # Update profile if changed
        engine = self._get_engine(profile)

        # Filter source
        source_id = None if source_filter == "All Sources" else source_filter

        try:
            # Query the engine
            response = engine.query(question=message, source_id=source_id)

            # Format sources for display
            if response.chunks_used:
                sources_html = self._format_sources_html(response)
            else:
                sources_html = ""

            # Add to history
            history = history + [(message, response.answer)]

            yield history, sources_html

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            history = history + [(message, error_msg)]
            yield history, ""

    def _format_sources_html(self, response) -> str:
        """Format sources as HTML for display."""
        if not response.chunks_used:
            return ""

        # Group by source
        source_info = {}
        for chunk in response.chunks_used:
            src = chunk.source_name
            if src not in source_info:
                source_info[src] = {"pages": set(), "score": 0}
            for p in range(chunk.start_page, chunk.end_page + 1):
                source_info[src]["pages"].add(p)
            source_info[src]["score"] = max(source_info[src]["score"], chunk.relevance_score)

        html_parts = ["<div class='source-box'>", "<strong>📚 Sources Used:</strong><br>"]

        for src, info in source_info.items():
            pages = sorted(info["pages"])
            if len(pages) == 1:
                page_str = f"p. {pages[0]}"
            elif len(pages) <= 3:
                page_str = f"pp. {', '.join(map(str, pages))}"
            else:
                page_str = f"pp. {pages[0]}-{pages[-1]}"

            relevance = int(info["score"] * 100)
            html_parts.append(f"• <strong>{src}</strong> ({page_str}) - {relevance}% relevant<br>")

        html_parts.append("</div>")
        return "".join(html_parts)

    def build_ui(self) -> gr.Blocks:
        """Build the Gradio interface."""

        with gr.Blocks(
            title="MedEducation",
            css=CUSTOM_CSS,
            theme=gr.themes.Soft(
                primary_hue="blue",
                secondary_hue="slate",
            ),
        ) as app:
            # Header
            gr.Markdown(
                """
                # 🏥 MedEducation
                ### Medical Education RAG Assistant

                Ask questions about your medical textbooks. Answers include citations and are tailored to your practice level.
                """
            )

            with gr.Row():
                # Left column - Settings
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ Settings")

                    profile_dropdown = gr.Dropdown(
                        choices=self.get_profiles(),
                        value=self.default_profile,
                        label="Profile",
                        info="Tailors response depth and focus",
                    )

                    profile_info = gr.Markdown(
                        value=self.get_profile_info(self.default_profile),
                        label="Profile Details",
                    )

                    source_dropdown = gr.Dropdown(
                        choices=self.get_sources(),
                        value="All Sources",
                        label="Source Filter",
                        info="Filter to specific textbook",
                    )

                    gr.Markdown(
                        """
                        ---
                        ### 📖 Quick Tips
                        - Ask detailed clinical questions
                        - Request differentials, dosages, mnemonics
                        - Specify patient scenarios for context
                        - All answers include page citations
                        """
                    )

                # Right column - Chat
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="Conversation",
                        height=500,
                        show_copy_button=True,
                        render_markdown=True,
                        avatar_images=(None, "🏥"),
                    )

                    sources_display = gr.HTML(
                        label="Sources",
                        visible=True,
                    )

                    with gr.Row():
                        msg_input = gr.Textbox(
                            placeholder="Ask a medical education question...",
                            label="Your Question",
                            scale=4,
                            lines=2,
                        )
                        submit_btn = gr.Button("Ask", variant="primary", scale=1)

                    with gr.Row():
                        clear_btn = gr.Button("🗑️ Clear Chat", size="sm")
                        example_btn = gr.Button("💡 Example Question", size="sm")

            # Example questions
            gr.Markdown("### 💡 Example Questions")
            gr.Examples(
                examples=[
                    ["What is the pathophysiology of cardiogenic shock and how does management differ from distributive shock?"],
                    ["Explain the RUSH exam protocol for undifferentiated hypotension"],
                    ["What are the indications and dosing for push-dose epinephrine vs norepinephrine?"],
                    ["Compare the presentation of tension pneumothorax vs massive hemothorax in trauma"],
                    ["What are the key considerations for RSI at altitude during flight transport?"],
                ],
                inputs=msg_input,
            )

            # Footer
            gr.Markdown(
                """
                ---
                <center>
                <small>MedEducation RAG Assistant | For educational purposes only | Always verify with current protocols</small>
                </center>
                """
            )

            # Event handlers
            def on_profile_change(profile):
                return self.get_profile_info(profile)

            profile_dropdown.change(
                fn=on_profile_change,
                inputs=[profile_dropdown],
                outputs=[profile_info],
            )

            def on_submit(message, history, source, profile):
                for result in self.chat(message, history, source, profile):
                    yield result[0], result[1], ""

            submit_btn.click(
                fn=on_submit,
                inputs=[msg_input, chatbot, source_dropdown, profile_dropdown],
                outputs=[chatbot, sources_display, msg_input],
            )

            msg_input.submit(
                fn=on_submit,
                inputs=[msg_input, chatbot, source_dropdown, profile_dropdown],
                outputs=[chatbot, sources_display, msg_input],
            )

            clear_btn.click(
                fn=lambda: ([], ""),
                outputs=[chatbot, sources_display],
            )

            def load_example():
                return "What are the signs and symptoms of tension pneumothorax and how should it be managed in the prehospital setting?"

            example_btn.click(
                fn=load_example,
                outputs=[msg_input],
            )

        return app


def create_app(
    vectordb_path: str = None,
    config_path: str = None,
    profile: str = "flight_critical_care",
) -> gr.Blocks:
    """Create the Gradio app.

    Args:
        vectordb_path: Path to vector database.
        config_path: Path to config file.
        profile: Default user profile.

    Returns:
        Gradio Blocks app.
    """
    # Find project root
    cwd = Path.cwd()
    project_root = cwd
    for parent in [cwd] + list(cwd.parents):
        if (parent / "pyproject.toml").exists() or (parent / "config").exists():
            project_root = parent
            break

    vectordb_path = vectordb_path or str(project_root / "data" / "vectordb")
    config_path = config_path or str(project_root / "config" / "sources.yaml")

    ui = MedEducationUI(
        vectordb_path=vectordb_path,
        config_path=config_path,
        default_profile=profile,
    )

    return ui.build_ui()


def launch(
    host: str = "0.0.0.0",
    port: int = 7860,
    share: bool = False,
    **kwargs,
):
    """Launch the web UI.

    Args:
        host: Host to bind to.
        port: Port to listen on.
        share: Create a public Gradio link.
        **kwargs: Additional arguments for gr.Blocks.launch()
    """
    app = create_app()
    app.launch(
        server_name=host,
        server_port=port,
        share=share,
        **kwargs,
    )


if __name__ == "__main__":
    launch()

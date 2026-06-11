import os
import logging
from flask import Flask, render_template, request, jsonify
from src.main import initialize_rag_system

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_app():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    template_folder = os.path.join(root_dir, 'templates')
    static_folder = os.path.join(root_dir, 'static')

    app = Flask(
        __name__,
        template_folder=template_folder,
        static_folder=static_folder
    )
    app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

    @app.before_request
    def init_rag():
        if request.path.startswith('/static'):
            return None

        if not hasattr(app, 'rag_pipeline') or app.rag_pipeline is None:
            logger.info('Initializing RAG pipeline for web app...')
            app.rag_pipeline = initialize_rag_system()
            logger.info('RAG pipeline initialized.')

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/chat', methods=['POST'])
    def chat():
        payload = request.get_json(force=True)
        question = payload.get('question', '').strip()
        history = payload.get('history', [])

        if not question:
            return jsonify({'error': 'Question cannot be empty.'}), 400

        try:
            chat_history = []
            for item in history:
                user_text = item.get('user', '')
                assistant_text = item.get('assistant', '')
                if user_text and assistant_text:
                    chat_history.append((user_text, assistant_text))

            result = app.rag_pipeline.chat(question, chat_history)
            return jsonify({
                'answer': result.get('answer', ''),
                'sources': [
                    {
                        'title': src['metadata'].get('name', src['metadata'].get('drug_name', 'unknown')),
                        'metadata': src['metadata'],
                        'content': src['content']
                    }
                    for src in result.get('sources', [])
                ]
            })
        except Exception as exc:
            logger.error(f'Web chat error: {exc}')
            return jsonify({'error': str(exc)}), 500

    return app


if __name__ == '__main__':
    web_app = create_app()
    web_app.run(host='127.0.0.1', port=5000, debug=False)

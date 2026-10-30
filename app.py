import os
import gradio as gr
from dotenv import load_dotenv
from google import genai
from PIL import Image

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=gemini_api_key)

SYSTEM_PROMPT = """
You are an expert AI prompt engineer for Image-to-Video generation.
Analyze the uploaded image and user movement request.

Generate a single copy-paste prompt formatted exactly like this:

添付の画像を元に、以下の指示に従って破綻のないショートアニメーション（動画）を生成してください。

[Core Motion]: (Concise English motion prompt)
[Visual Consistency]: (Key visual traits to preserve from the image)
[Camera & Scene]: Static camera, lock background elements and lighting.
[Strict Constraints]: No facial distortion, no extra limbs, preserve textures.

Output ONLY this formatted text block without markdown headers or greetings.
"""

def generate_video_guide(image_path: str, instruction: str, motion_type: str):
    if not image_path:
        raise gr.Error("画像をアップロードしてください。")
    if not instruction.strip():
        instruction = "自然に息をするように揺れる、愛らしい微小な動き"

    try:
        img = Image.open(image_path)
        user_message = f"Movement: {instruction}, Style: {motion_type}"

        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[img, SYSTEM_PROMPT, user_message],
        )
        return response.text.strip()
    except Exception as e:
        raise gr.Error(f"APIエラー: {e}")

with gr.Blocks(title="nui-to-anime") as demo:
    gr.Markdown("# 🧸 nui-to-anime Studio")
    gr.Markdown("画像をアップロードして指示を入力すると、チャットAIへそのまま渡せるプロンプトを作成します。")

    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="filepath", label="📷 ぬいぐるみの写真", sources=["upload"])
            instruction = gr.Textbox(
                label="💡 やりたい動き（日本語）",
                placeholder="例: 右手を小さく振る",
                value="右手を小さく振って挨拶する"
            )
            motion_type = gr.Radio(
                label="動きのテイスト",
                choices=["控えめ・微小動作（破綻防止）", "元気なアクション", "ストップモーション（コマ撮り風）", "ふわふわ・浮遊感"],
                value="控えめ・微小動作（破綻防止）"
            )
            submit_btn = gr.Button("🚀 コピペ用プロンプトを生成する", variant="primary")

        with gr.Column(scale=1):
            unified_output = gr.Textbox(
                label="📋 チャットAI貼り付け用テキスト（画像と一緒に送信）",
                lines=12,
                interactive=False
            )

    submit_btn.click(
        fn=generate_video_guide,
        inputs=[input_img, instruction, motion_type],
        outputs=[unified_output],
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=port,
        theme=gr.themes.Soft()
    )

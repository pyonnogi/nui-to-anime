import os
import shutil
import tempfile
from dotenv import load_dotenv
from google import genai
import gradio as gr
from gradio_client import Client, handle_file
from PIL import Image

load_dotenv()

# クライアント初期化
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
hf_token = os.getenv("HF_TOKEN")

def generate_motion_prompt(image_path: str, instruction: str) -> str:
    """Gemini 2.5 Flash で動画指示文を最適化"""
    img = Image.open(image_path)
    prompt = f"""
    The user wants to animate this plushie/toy.
    User instruction: {instruction}
    
    Output ONLY a concise, high-quality prompt in English for an Image-to-Video diffusion model.
    Requirements:
    - Keep background static.
    - Subtle, natural, cute movement.
    - Output just the raw prompt text without extra explanation.
    """
    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[img, prompt],
    )
    return response.text.strip()

def process_animation(image_data, instruction: str):
    if image_data is None:
        raise gr.Error("画像をアップロードしてください。")
    if not instruction:
        instruction = "Subtle wiggle and breathing movement"

    image_path = image_data["background"] if isinstance(image_data, dict) else image_data

    # 1. Geminiでプロンプト生成
    motion_prompt = generate_motion_prompt(image_path, instruction)

    # 2. 無料の動画生成Space API呼び出し
    client = Client("multimodalart/wan2-1-fast", token=hf_token)
    result = client.predict(
        image=handle_file(image_path),
        prompt=motion_prompt,
        negative_prompt="camera movement, distorted face, low quality",
        api_name="/generate_video",
    )

    if isinstance(result, (list, tuple)):
        result = result[0]

    # 一時領域に退避してストレージ保持を防ぐ
    temp_dir = tempfile.gettempdir()
    temp_output_path = os.path.join(temp_dir, f"temp_{os.path.basename(result)}")
    shutil.copy(result, temp_output_path)

    return temp_output_path, motion_prompt

# UI画面の構築
with gr.Blocks(title="nui-to-anime") as demo:
    gr.Markdown("# 🧸 nui-to-anime（ぬいぐるみアニメーション変換ツール）")
    gr.Markdown("写真をアップロードして指示を入力すると、動画をその場で生成して返します。")

    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="filepath", label="写真アップロード", sources=["upload"])
            instruction = gr.Textbox(
                label="動きの指示",
                placeholder="例: 手を小さく振る、息をするように揺れる",
                value="手を小さく振る"
            )
            submit_btn = gr.Button("アニメーション生成", variant="primary")
        with gr.Column():
            output_vid = gr.Video(label="生成結果", autoplay=True, loop=True)
            output_prompt = gr.Textbox(label="最適化プロンプト", interactive=False)

    submit_btn.click(
        fn=process_animation,
        inputs=[input_img, instruction],
        outputs=[output_vid, output_prompt],
    )

if __name__ == "__main__":
    # Render環境で割り振られるPORTを取得（未指定時は7860）
    port = int(os.environ.get("PORT", 7860))
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=port
    )

import os
import gradio as gr
from dotenv import load_dotenv
from google import genai
from PIL import Image

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=gemini_api_key)

SYSTEM_PROMPT = """
You are an expert AI prompt engineer for Image-to-Video generation using Gemini.
Analyze the uploaded plushie/figure image and user parameters to generate ONE production-ready copy-paste prompt.

Format the output exactly as follows:

添付の画像を元に、以下の指示に従って破綻のないショートアニメーション（動画）を生成してください。

[Core Motion]: (Concise English motion prompt based on user's instruction and speed)
[Visual Consistency]: (Preserve key colors, plush/fabric texture, and original form)
[Face & Expression]: (Rules on whether eyes/mouth remain completely static or can blink/shift)
[Camera & Angle]: (Static camera or specific camera motion as selected)
[Continuity & Scene Flow]: (Stand-alone generation vs. continuing sequence/scene transition)
[Strict Constraints]: No unexpected morphing, no extra limbs, no background glitching.

Output ONLY this unified text block. No preamble, no postscript.
"""

def generate_video_guide(
    image_path: str,
    instruction: str,
    camera_option: str,
    continuity_option: str,
    face_option: str,
    speed_option: str
):
    if not image_path:
        raise gr.Error("画像をアップロードしてください。")
    if not instruction.strip():
        instruction = "自然に息をするように揺れる、愛らしい微小な動き"

    try:
        img = Image.open(image_path)
        user_message = f"""
        User Parameters:
        - Movement: {instruction}
        - Camera Angle: {camera_option}
        - Continuity: {continuity_option}
        - Face/Expression Lock: {face_option}
        - Motion Speed: {speed_option}
        """

        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[img, SYSTEM_PROMPT, user_message],
        )
        return response.text.strip()
    except Exception as e:
        raise gr.Error(f"APIエラー: {e}")

with gr.Blocks(title="nui-to-anime") as demo:
    gr.Markdown("# 🧸 nui-to-anime Studio")
    gr.Markdown("写真と設定を選ぶだけで、チャットAIにコピペするだけの統合プロンプトを作成します。")

    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="filepath", label="📷 ぬいぐるみの写真", sources=["upload"])
            instruction = gr.Textbox(
                label="💡 やりたい動き（日本語）",
                placeholder="例: 右手を小さく振る",
                value="右手を小さく振って挨拶する"
            )
            camera_option = gr.Radio(
                label="🎥 アングル・カメラワーク",
                choices=["アングル完全固定（推奨）", "ゆっくりズームイン", "アングル変化を許可"],
                value="アングル完全固定（推奨）"
            )
            continuity_option = gr.Radio(
                label="🎞️ 前回の生成結果の反映",
                choices=["新規作成（単一シーン）", "前回の続きを反映（場面転換や急な展開を許可）"],
                value="新規作成（単一シーン）"
            )
            face_option = gr.Radio(
                label="👀 表情・顔パーツの固定度",
                choices=["表情完全固定（変形・瞬き禁止）", "瞬きや軽い表情変化を許可"],
                value="表情完全固定（変形・瞬き禁止）"
            )
            speed_option = gr.Radio(
                label="⏱️ 再生速度・テンポ感",
                choices=["ゆったり・スロー（ほのぼの）", "通常速度", "小刻み・ちょこまか（コミカル）"],
                value="ゆったり・スロー（ほのぼの）"
            )
            submit_btn = gr.Button("🚀 コピペ用プロンプトを生成する", variant="primary")

        with gr.Column(scale=1):
            unified_output = gr.Textbox(
                label="📋 チャットAI貼り付け用テキスト（画像と一緒に送信）",
                lines=14,
                interactive=False
            )

    submit_btn.click(
        fn=generate_video_guide,
        inputs=[
            input_img,
            instruction,
            camera_option,
            continuity_option,
            face_option,
            speed_option
        ],
        outputs=[unified_output],
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=port,
        theme=gr.themes.Soft()
    )

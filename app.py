import os
import gradio as gr
from dotenv import load_dotenv
from google import genai
from PIL import Image

load_dotenv()

# Gemini クライアント初期化
gemini_api_key = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=gemini_api_key)

SYSTEM_PROMPT = """
You are an expert AI prompt engineer specializing in Image-to-Video diffusion models (such as Runway Gen-3, Luma Dream Machine, Kling, Sora, and Wan2.1).
Analyze the uploaded image of the plushie/figure and the user's movement intent.

Generate:
1. **Target Movement Analysis**: Briefly break down which parts should move (limbs, head, breathing effect).
2. **Optimized English Prompt (Diffusion-ready)**: A precise, high-density English prompt describing the subject, micro-motions, static background, and lighting preservation. Avoid buzzwords like 'photorealistic'.
3. **Camera & Motion Settings Guide**: Recommendations for Motion strength (1-10), Camera movement (e.g., Static / Slow push-in), and FPS.
4. **Negative Prompt**: Standard negative prompts to avoid distortion and unnatural warping.

Keep the tone practical, professional, and easy to copy-paste.
"""

def generate_video_guide(image_path: str, instruction: str, motion_type: str):
    if not image_path:
        raise gr.Error("画像をアップロードしてください。")
    if not instruction.strip():
        instruction = "自然に息をするように揺れる、愛らしい微小な動き"

    try:
        img = Image.open(image_path)
        
        user_message = f"""
        [User Instruction]: {instruction}
        [Movement Preference]: {motion_type}
        
        Please generate the production-ready Image-to-Video prompts based on the visual features of this plushie/figure.
        """

        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[img, SYSTEM_PROMPT, user_message],
        )

        full_text = response.text

        # コピー用の英語プロンプト抽出
        extract_prompt = f"""
        Extract ONLY the primary diffusion-ready English prompt from the text below. 
        Do not include markdown tags, titles, or notes. Output just the raw text:
        
        {full_text}
        """
        short_prompt_resp = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[extract_prompt],
        )
        pure_prompt = short_prompt_resp.text.strip().replace('"', '')

        return full_text, pure_prompt

    except Exception as e:
        print(f"[Gemini Error]: {e}")
        raise gr.Error(f"Gemini API呼び出し中にエラーが発生しました: {e}")

# Gradio UI構築（Gradio 6.x 互換）
with gr.Blocks(title="nui-to-anime") as demo:
    gr.Markdown("# 🧸 nui-to-anime Studio")
    gr.Markdown(
        "ぬいぐるみの写真をアップロードし、やりたい動きを日本語で入力してください。\n"
        "Geminiが画像の特徴を解析し、動画生成AIで破綻なく動かすための**最適化プロンプト・カメラ指示・パラメータ設定**を作成します。"
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="filepath", label="📷 ぬいぐるみの写真", sources=["upload"])
            instruction = gr.Textbox(
                label="💡 やりたい動き（日本語）",
                placeholder="例: 右手を小さく振る、首を少し傾げて瞬きする",
                value="右手を小さく振って挨拶する"
            )
            motion_type = gr.Radio(
                label="動きのテイスト",
                choices=["控えめ・微小動作（破綻防止）", "元気なアクション", "ストップモーション（コマ撮り風）", "ふわふわ・浮遊感"],
                value="控えめ・微小動作（破綻防止）"
            )
            submit_btn = gr.Button("🚀 最適化プロンプトを生成する", variant="primary")

        with gr.Column(scale=1):
            copy_prompt = gr.Textbox(
                label="📋 コピペ用プロンプト（動画AI入力用）",
                interactive=False,
                placeholder="生成された英語プロンプトがここに表示されます。"
            )
            full_guide = gr.Markdown(
                value="*ここにGeminiによる詳細な解析結果とパラメータ設定が表示されます。*"
            )

    submit_btn.click(
        fn=generate_video_guide,
        inputs=[input_img, instruction, motion_type],
        outputs=[full_guide, copy_prompt],
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=port,
        theme=gr.themes.Soft()
    )

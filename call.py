import discord
from discord.ext import commands
from test import get_ai_response
import os
from dotenv import load_dotenv

load_dotenv()

# 2. Lấy giá trị bằng os.getenv
discord_bot_token = os.getenv("discord_bot_token")
# 1. Cấu hình Intents (Quyền hạn)
intents = discord.Intents.default()
intents.message_content = True  # Quan trọng: Cho phép đọc nội dung tin nhắn

# 2. Khởi tạo Bot
bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    print(f'✅ Bot đã online với tên: {bot.user}')

@bot.command()
async def ask(ctx, *, question):
    async with ctx.typing(): # Hiệu ứng Bot đang gõ
        # Gọi hàm từ file test.py
        result = await get_ai_response("chỉ trả lời ngắn gọn trong 200 từ:" + question)
        
        # Gửi trả lời lại Discord
        await ctx.send(f"🤖 **AI trả lời:**\n{result}")


bot.run(discord_bot_token)
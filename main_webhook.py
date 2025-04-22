
import os
import joblib
import numpy as np
from aiogram import Bot, Dispatcher, types
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from fastapi import FastAPI, Request
from aiogram.types import Update
from aiogram.utils.executor import start_webhook

API_TOKEN = os.getenv("BOT_TOKEN")
WEBHOOK_PATH = "/webhook"
WEBHOOK_URL = os.getenv("RENDER_EXTERNAL_URL", "") + WEBHOOK_PATH
WEBAPP_HOST = "0.0.0.0"
WEBAPP_PORT = int(os.getenv("PORT", 8000))

bot = Bot(token=API_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)
app = FastAPI()

class Form(StatesGroup):
    brand = State()
    model = State()
    year = State()
    mileage = State()
    engine = State()
    fuel = State()
    country = State()

@dp.message_handler(commands="start")
async def start(message: types.Message):
    await message.reply("Привіт! Введи марку авто (наприклад: Toyota):")
    await Form.brand.set()

@dp.message_handler(state=Form.brand)
async def brand(message: types.Message, state: FSMContext):
    await state.update_data(brand=message.text)
    await message.reply("Введи модель:")
    await Form.model.set()

@dp.message_handler(state=Form.model)
async def model(message: types.Message, state: FSMContext):
    await state.update_data(model=message.text)
    await message.reply("Рік випуску:")
    await Form.year.set()

@dp.message_handler(state=Form.year)
async def year(message: types.Message, state: FSMContext):
    await state.update_data(year=int(message.text))
    await message.reply("Пробіг (тис. км):")
    await Form.mileage.set()

@dp.message_handler(state=Form.mileage)
async def mileage(message: types.Message, state: FSMContext):
    await state.update_data(mileage=int(message.text))
    await message.reply("Обʼєм двигуна:")
    await Form.engine.set()

@dp.message_handler(state=Form.engine)
async def engine(message: types.Message, state: FSMContext):
    await state.update_data(engine=float(message.text))
    await message.reply("Тип палива (petrol/diesel/hybrid/electric):")
    await Form.fuel.set()

@dp.message_handler(state=Form.fuel)
async def fuel(message: types.Message, state: FSMContext):
    await state.update_data(fuel=message.text.lower())
    await message.reply("Країна (UA, EU, USA):")
    await Form.country.set()

@dp.message_handler(state=Form.country)
async def country(message: types.Message, state: FSMContext):
    await state.update_data(country=message.text.upper())
    data = await state.get_data()

    try:
        model, columns = joblib.load("car_value_model.pkl")
        input_dict = {
            'brand': data['brand'],
            'model': data['model'],
            'age': 2025 - data['year'],
            'mileage': data['mileage'],
            'engine': data['engine'],
            'fuel': data['fuel'],
            'country': data['country']
        }

        input_vector = {col: 0 for col in columns}
        for key, value in input_dict.items():
            colname = f"{key}_{value}" if f"{key}_{value}" in columns else key
            if colname in input_vector:
                input_vector[colname] = value if isinstance(value, (int, float)) else 1

        X_input = np.array([list(input_vector.values())])
        price = model.predict(X_input)[0]
        await message.reply(f"Орієнтовна вартість авто: ${round(price, 2)}")
    except Exception as e:
        await message.reply("Вибач, не вдалося виконати оцінку. Спробуй ще раз.")
    await state.finish()

@app.on_event("startup")
async def on_startup():
    await bot.set_webhook(WEBHOOK_URL)

@app.on_event("shutdown")
async def on_shutdown():
    await bot.delete_webhook()

@app.post(WEBHOOK_PATH)
async def webhook(request: Request):
    data = await request.json()
    update = Update.to_object(data)
    await dp.process_update(update)
    return {"ok": True}

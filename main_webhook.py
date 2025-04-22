
import os
import joblib
import numpy as np
from fastapi import FastAPI, Request
from aiogram import Bot, Dispatcher, types
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.types import Update

API_TOKEN = os.getenv("BOT_TOKEN")
WEBHOOK_PATH = "/webhook"
WEBHOOK_URL = os.getenv("RENDER_EXTERNAL_URL", "") + WEBHOOK_PATH
WEBAPP_HOST = "0.0.0.0"
WEBAPP_PORT = int(os.getenv("PORT", 10000))

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

@app.on_event("startup")
async def startup():
    await bot.set_webhook(WEBHOOK_URL)

@app.on_event("shutdown")
async def shutdown():
    await bot.delete_webhook()
    session = await bot.get_session()
    await session.close()

@app.post(WEBHOOK_PATH)
async def telegram_webhook(request: Request):
    data = await request.json()
    update = Update.to_object(data)
    await dp.process_update(update)
    return {"ok": True}

@dp.message_handler(commands="start")
async def cmd_start(message: types.Message, state: FSMContext):
    await bot.send_message(message.chat.id, "Привіт! Введи марку авто (наприклад: Toyota):")
    await state.set_state(Form.brand.state)

@dp.message_handler(state=Form.brand)
async def form_brand(message: types.Message, state: FSMContext):
    await state.update_data(brand=message.text)
    await bot.send_message(message.chat.id, "Введи модель авто:")
    await state.set_state(Form.model.state)

@dp.message_handler(state=Form.model)
async def form_model(message: types.Message, state: FSMContext):
    await state.update_data(model=message.text)
    await bot.send_message(message.chat.id, "Введи рік випуску:")
    await state.set_state(Form.year.state)

@dp.message_handler(state=Form.year)
async def form_year(message: types.Message, state: FSMContext):
    await state.update_data(year=int(message.text))
    await bot.send_message(message.chat.id, "Введи пробіг (тис. км):")
    await state.set_state(Form.mileage.state)

@dp.message_handler(state=Form.mileage)
async def form_mileage(message: types.Message, state: FSMContext):
    await state.update_data(mileage=int(message.text))
    await bot.send_message(message.chat.id, "Введи обʼєм двигуна:")
    await state.set_state(Form.engine.state)

@dp.message_handler(state=Form.engine)
async def form_engine(message: types.Message, state: FSMContext):
    await state.update_data(engine=float(message.text))
    await bot.send_message(message.chat.id, "Введи тип палива (petrol/diesel/electric/hybrid):")
    await state.set_state(Form.fuel.state)

@dp.message_handler(state=Form.fuel)
async def form_fuel(message: types.Message, state: FSMContext):
    await state.update_data(fuel=message.text.lower())
    await bot.send_message(message.chat.id, "Введи країну (UA, EU, USA):")
    await state.set_state(Form.country.state)

@dp.message_handler(state=Form.country)
async def form_country(message: types.Message, state: FSMContext):
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
        await bot.send_message(message.chat.id, f"Орієнтовна вартість авто: ${round(price, 2)}")
    except Exception as e:
        await bot.send_message(message.chat.id, f"Сталась помилка: {e}")

    await state.finish()

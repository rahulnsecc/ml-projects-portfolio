import sys
import traceback
from datetime import datetime
from http import HTTPStatus
from aiohttp import web
from aiohttp.web import Request, Response, json_response
from botbuilder.core import ConversationState, MemoryStorage, TurnContext, UserState
from botbuilder.core.integration import aiohttp_error_middleware
from botbuilder.integration.aiohttp import CloudAdapter, ConfigurationBotFrameworkAuthentication
from botbuilder.schema import Activity, ActivityTypes
from config import DefaultConfig
from user_profile_dialog import UserProfileDialog
from sop_bot import SOPBOT

CONFIG = DefaultConfig()

ADAPTER = CloudAdapter(ConfigurationBotFrameworkAuthentication(CONFIG))

async def on_error(context: TurnContext, error: Exception):
    print(f"[on_turn_error] unhandled error: {error}", file=sys.stderr)
    traceback.print_exc()
    await context.send_activity("The bot encountered an error or bug.")
    await context.send_activity("To continue to run this bot, please fix the bot source code.")
    if context.activity.channel_id == "emulator":
        trace_activity = Activity(
            label="TurnError",
            name="on_turn_error Trace",
            timestamp=datetime.utcnow(),
            type=ActivityTypes.trace,
            value=f"{error}",
            value_type="https://www.botframework.com/schemas/error"
        )
        await context.send_activity(trace_activity)
    await CONVERSATION_STATE.delete(context)

ADAPTER.on_turn_error = on_error

MEMORY = MemoryStorage()
CONVERSATION_STATE = ConversationState(MEMORY)
USER_STATE = UserState(MEMORY)
DIALOG = UserProfileDialog(USER_STATE)
BOT = SOPBOT(CONVERSATION_STATE, USER_STATE, DIALOG)

async def messages(req: Request) -> Response:
    print("[messages] Handling request")
    if "application/json" in req.headers["Content-Type"]:
        body = await req.json()
        print(f"[messages] Received request body: {body}")
        auth_header = req.headers["Authorization"] if "Authorization" in req.headers else ""
        activity = Activity().deserialize(body)
        response = await ADAPTER.process_activity(auth_header, activity, BOT.on_turn)
        if response:
            return json_response(data=response.body, status=response.status)
    return Response(status=HTTPStatus.OK)

APP = web.Application(middlewares=[aiohttp_error_middleware])
APP.router.add_post("/api/messages", messages)

if __name__ == "__main__":
    web.run_app(APP, host="localhost", port=CONFIG.PORT)

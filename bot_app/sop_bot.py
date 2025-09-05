from botbuilder.core import TurnContext
from botbuilder.dialogs import Dialog
from botbuilder.core import ConversationState, UserState
from dialog_helper import DialogHelper

class SOPBOT:
    def __init__(self, conversation_state: ConversationState, user_state: UserState, dialog: Dialog):
        self.conversation_state = conversation_state
        self.user_state = user_state
        self.dialog = dialog

    async def on_turn(self, turn_context: TurnContext):
        print(f"[on_turn] Activity type: {turn_context.activity.type}")
        if turn_context.activity.type == "message":
            await DialogHelper.run_dialog(self.dialog, turn_context, self.conversation_state.create_property("DialogState"))
        elif turn_context.activity.type == "conversationUpdate":
            for member in turn_context.activity.members_added:
                if member.id != turn_context.activity.recipient.id:
                    await turn_context.send_activity("Welcome to the SOP BOT!")
        await self.conversation_state.save_changes(turn_context)
        await self.user_state.save_changes(turn_context)

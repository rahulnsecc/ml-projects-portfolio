import json
import os
import csv
from datetime import datetime
from uuid import uuid4
from dotenv import load_dotenv
from botbuilder.dialogs import (
    ComponentDialog,
    WaterfallDialog,
    WaterfallStepContext,
    DialogTurnResult,
    TextPrompt,
    PromptOptions,
    PromptValidatorContext,
)
from botbuilder.core import MessageFactory
from config import DefaultConfig
from process_user_request_dao import ProcessUserRequestDAO
from azure.storage.blob import BlobServiceClient
from clu_utility import CLUUtility

load_dotenv()

class UserProfileDialog(ComponentDialog):
    def __init__(self, user_state):
        super(UserProfileDialog, self).__init__(UserProfileDialog.__name__)
        self.user_profile_accessor = user_state.create_property("UserProfile")
        self.add_dialog(TextPrompt(TextPrompt.__name__))
        self.add_dialog(TextPrompt("DatePrompt", self.date_validator))
        self.add_dialog(
            WaterfallDialog(
                WaterfallDialog.__name__,
                [
                    self.initial_step,
                    self.process_input_step,
                    self.prompt_for_required_fields_step,
                    self.handle_required_fields_step,  # New step added
                    self.prompt_for_optional_fields_step,
                    self.handle_optional_fields_step,
                    self.confirm_step,
                    self.handle_confirmation_step,
                ],
            )
        )
        self.initial_dialog_id = WaterfallDialog.__name__
        self.blob_service_client = BlobServiceClient.from_connection_string(DefaultConfig.AZURE_STORAGE_CONNECTION_STRING)
        self.container_name = DefaultConfig.AZURE_STORAGE_CONTAINER_NAME
        self.clu_utility = CLUUtility()

    @staticmethod
    async def date_validator(prompt_context: PromptValidatorContext) -> bool:
        try:
            date_str = prompt_context.recognized.value
            datetime.strptime(date_str, DefaultConfig.DATE_FORMATS[0])
            return True
        except ValueError:
            return False

    async def initial_step(self, step_context: WaterfallStepContext) -> DialogTurnResult:
        return await step_context.prompt(
            TextPrompt.__name__,
            PromptOptions(prompt=MessageFactory.text(DefaultConfig.MESSAGES["prompt_request"])),
        )

    async def process_input_step(self, step_context: WaterfallStepContext) -> DialogTurnResult:
        user_request = step_context.result.lower()
        analysis_result = self.clu_utility.analyze_input(user_request)

        intent = analysis_result["topIntent"]
        entities = analysis_result["entities"]
        key_mappings = DefaultConfig.KEY_MAPPINGS

        if intent not in DefaultConfig.REPORT_CONFIG:
            await step_context.context.send_activity("I'm sorry, I didn't understand that request. Please try again.")
            return await step_context.replace_dialog(UserProfileDialog.__name__)

        mapped_entities = step_context.values.get("entities", {})
        for entity in entities:
            category = entity["category"]
            if category in key_mappings:
                mapped_key = key_mappings[category]
                mapped_entities[mapped_key] = entity["text"]

        step_context.values["intent"] = intent
        step_context.values["entities"] = mapped_entities

        report_config = DefaultConfig.REPORT_CONFIG[intent]
        missing_required_fields = [field for field in report_config["required_fields"] if field not in mapped_entities]

        if missing_required_fields:
            step_context.values["missing_required_fields"] = missing_required_fields
            return await step_context.next(None)

        return await step_context.next(None)

    async def prompt_for_required_fields_step(self, step_context: WaterfallStepContext) -> DialogTurnResult:
        intent = step_context.values["intent"]
        entities = step_context.values["entities"]
        report_config = DefaultConfig.REPORT_CONFIG[intent]
        missing_required_fields = step_context.values.get("missing_required_fields", [])

        if missing_required_fields:
            missing_field = missing_required_fields[0]
            return await step_context.prompt(
                TextPrompt.__name__,
                PromptOptions(prompt=MessageFactory.text(f"Please provide the {missing_field.replace('_', ' ')}."))
            )

        start_date = entities.get("start_date")
        end_date = entities.get("end_date")
        print("Start Date:", start_date)
        print("End Date:", end_date)

        if start_date and end_date and not self.validate_dates(start_date, end_date):
            await step_context.context.send_activity("The dates provided are not valid. Please enter valid dates.")
            return await step_context.replace_dialog(UserProfileDialog.__name__)

        step_context.values["start_date"] = start_date
        step_context.values["end_date"] = end_date

        return await step_context.next(None)

    async def handle_required_fields_step(self, step_context: WaterfallStepContext) -> DialogTurnResult:
        missing_required_fields = step_context.values.get("missing_required_fields", [])
        entities = step_context.values["entities"]

        if missing_required_fields:
            field = missing_required_fields.pop(0)
            entities[field] = step_context.result
            step_context.values["entities"] = entities
            step_context.values["missing_required_fields"] = missing_required_fields

            if field in ["start_date", "end_date"]:
                start_date = entities.get("start_date")
                end_date = entities.get("end_date")
                print("Start Date:", start_date)
                print("End Date:", end_date)

                if start_date and end_date and not self.validate_dates(start_date, end_date):
                    await step_context.context.send_activity("The dates provided are not valid. Please enter valid dates.")
                    return await step_context.replace_dialog(UserProfileDialog.__name__)

                step_context.values["start_date"] = start_date
                step_context.values["end_date"] = end_date

            if missing_required_fields:
                next_field = missing_required_fields[0]
                return await step_context.prompt(
                    TextPrompt.__name__,
                    PromptOptions(prompt=MessageFactory.text(f"Please provide the {next_field.replace('_', ' ')}."))
                )

        return await step_context.next(None)

    def validate_dates(self, start_date: str, end_date: str) -> bool:
        def parse_date(date_str):
            for date_format in DefaultConfig.DATE_FORMATS:
                try:
                    return datetime.strptime(date_str, date_format)
                except ValueError:
                    continue
            return None

        start_date_dt = parse_date(start_date)
        end_date_dt = parse_date(end_date)

        if not start_date_dt or not end_date_dt:
            return False

        if start_date_dt > end_date_dt or start_date_dt > datetime.now() or end_date_dt > datetime.now():
            return False

        return True

    async def prompt_for_optional_fields_step(self, step_context: WaterfallStepContext) -> DialogTurnResult:
        if "prompted_for_optional" not in step_context.values:
            step_context.values["prompted_for_optional"] = True
            entities = step_context.values["entities"]
            optional_fields = ", ".join(DefaultConfig.REPORT_CONFIG[step_context.values["intent"]]["optional_fields"])
            provided_fields = ", ".join([f"{key}: {value}" for key, value in entities.items()])
            await step_context.context.send_activity(f"The following fields have been provided:\n{provided_fields}")
            return await step_context.prompt(
                TextPrompt.__name__,
                PromptOptions(prompt=MessageFactory.text(f"Would you like to provide optional fields ({optional_fields})? Please type 'yes' or 'no'.")),
            )
        else:
            return await step_context.next(step_context.result)

    async def handle_optional_fields_step(self, step_context: WaterfallStepContext) -> DialogTurnResult:
        user_input = step_context.result.lower()

        if user_input in ["yes", "no"]:
            if user_input == "yes":
                return await step_context.prompt(
                    TextPrompt.__name__,
                    PromptOptions(prompt=MessageFactory.text("Please provide the optional fields.")),
                )
            else:
                return await step_context.next(None)
        else:
            analysis_result = self.clu_utility.analyze_input(user_input)
            entities = analysis_result["entities"]
            key_mappings = DefaultConfig.KEY_MAPPINGS

            mapped_entities = step_context.values.get("entities", {})
            for entity in entities:
                category = entity["category"]
                if category in key_mappings:
                    mapped_key = key_mappings[category]
                    mapped_entities[mapped_key] = entity["text"]

            step_context.values["entities"] = mapped_entities

            return await step_context.next(None)

    async def confirm_step(self, step_context: WaterfallStepContext) -> DialogTurnResult:
        if step_context.values.get("prompted_for_optional"):
            if step_context.result:
                user_request = step_context.result.lower()
                analysis_result = self.clu_utility.analyze_input(user_request)
                entities = analysis_result["entities"]
                key_mappings = DefaultConfig.KEY_MAPPINGS

                mapped_entities = step_context.values.get("entities", {})
                for entity in entities:
                    category = entity["category"]
                    if category in key_mappings:
                        mapped_key = key_mappings[category]
                        mapped_entities[mapped_key] = entity["text"]

                step_context.values["entities"] = mapped_entities

        summary = "\n".join([f"{key.replace('_', ' ').title()}: {value}" for key, value in step_context.values["entities"].items()])
        await step_context.context.send_activity(f"Thank you. Here is the information you have provided:\n{summary}")
        return await step_context.prompt(
            TextPrompt.__name__,
            PromptOptions(prompt=MessageFactory.text("Is this information correct? Please type 'yes' or 'no'.")),
        )

    async def handle_confirmation_step(self, step_context: WaterfallStepContext) -> DialogTurnResult:
        try:
            user_input = step_context.result.lower()
            if user_input == "yes":
                await self.create_report(step_context)
                return await step_context.end_dialog()
            else:
                await step_context.context.send_activity("Let's try again.")
                return await step_context.replace_dialog(UserProfileDialog.__name__)
        except Exception as e:
            ticket_number = str(uuid4())
            await step_context.context.send_activity(f"An error occurred while processing your request. We have opened a support ticket with the following ticket number: {ticket_number}")
            return await step_context.end_dialog()

    async def create_report(self, step_context: WaterfallStepContext):
        try:
            intent = step_context.values["intent"]
            entities = step_context.values["entities"]
            process_user_request_dao = ProcessUserRequestDAO()
            start_date = entities.get("start_date")
            end_date = entities.get("end_date")
            store_number = entities.get("store_number")
            rx_number = entities.get("rx_number")

            print("Intent:", intent)
            print("Entities:", json.dumps(entities, indent=2))
            print("Start Date:", start_date)
            print("End Date:", end_date)
            print("Store Number:", store_number)
            print("RX Number:", rx_number)

            if not start_date or not end_date:
                raise ValueError("Missing required parameters for report generation.")

            data = process_user_request_dao.process_user_request(intent, start_date, end_date, store_number, rx_number)
            csv_data = [record.__dict__ for record in data]
            csv_file_path = os.path.join(DefaultConfig.TEMP_DIR, f"report_{uuid4()}.csv")

            with open(csv_file_path, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_data[0].keys())
                writer.writeheader()
                writer.writerows(csv_data)

            blob_client = self.blob_service_client.get_blob_client(container=self.container_name, blob=os.path.basename(csv_file_path))
            with open(csv_file_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)

            download_url = blob_client.url
            await step_context.context.send_activity(f"Your report has been generated. You can download it from the following link: {download_url}")

        except Exception as e:
            ticket_number = str(uuid4())
            await step_context.context.send_activity(f"An error occurred while creating your report. We have opened a support ticket with the following ticket number: {ticket_number}")

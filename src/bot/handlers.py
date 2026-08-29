import logging
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)
from telegram import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.error import BadRequest, TelegramError
from telegram.ext import ContextTypes

from src.agents.multimodal_agent import encode_image_as_data_url
from src.agents.orchestrator import (
    INTENT_KNOWLEDGE_QUESTION,
    INTENT_TRANSLATION,
    DocumentTurn,
    add_preference,
    clear_history,
    clear_preferences,
    get_history,
    get_preferences,
    process_document,
    process_message,
    remove_preference,
    tidy_preferences,
)
from src.bot.admin import LOG_LEVELS, is_admin
from src.bot.channel import channel_name, post_announcement
from src.bot.feedback import VERDICT_DOWN, VERDICT_UP, record_feedback
from src.bot.middleware import (
    KIND_PHOTO,
    KIND_TEXT,
    QUOTA_KINDS,
    DailyQuota,
    ValidationError,
    is_large_upload,
    validate_image_upload,
    validate_message_text,
)
from src.bot.transcript import format_transcript
from src.bot.typing_indicator import show_typing

logger = logging.getLogger(__name__)

ERROR_TIMEOUT = (
    "The AI service is taking too long to respond. Please try again in a moment."
)
ERROR_CONNECTION = (
    "I can't reach the AI service right now. Please try again in a few minutes."
)
ERROR_RATE_LIMIT = (
    "The AI service is currently overloaded. Please try again in a few minutes."
)
ERROR_GENERIC = (
    "Sorry, something went wrong while processing your question. "
    "Please try again later."
)
ERROR_NEEDS_OPERATOR = (
    "I can't use the AI service right now, and this is not something waiting "
    "will fix - the bot's operator has to sort it out. Sorry."
)
ERROR_OUTAGE = (
    "The AI service is having problems on its side right now. "
    "Please try again in a few minutes."
)
ERROR_TOO_LONG = (
    "That was too much for me to handle in one go. Please send a shorter "
    "message, or a photo of just the part you need."
)

# A 429 means either "slow down" or "your balance is gone", and only the
# error body tells them apart. A 400 is likewise several things at once.
QUOTA_EXHAUSTED_CODE = "insufficient_quota"
CONTEXT_LENGTH_CODE = "context_length_exceeded"


@dataclass(frozen=True)
class _Failure:
    """One way an AI call can fail: what the user reads, what the log says.

    Attributes:
        reply: The message sent to the user.
        log: Log line; its "%s" takes the description of the request.
        level: Level to log at. ERROR is what reaches the operator's chat.
    """

    reply: str
    log: str
    level: int = logging.WARNING


FAILURE_TIMEOUT = _Failure(ERROR_TIMEOUT, "LLM timeout for %s")
FAILURE_CONNECTION = _Failure(ERROR_CONNECTION, "LLM connection error for %s")
FAILURE_THROTTLED = _Failure(ERROR_RATE_LIMIT, "OpenAI rate limit hit for %s")
FAILURE_QUOTA = _Failure(
    ERROR_NEEDS_OPERATOR,
    "OpenAI credit exhausted, turning away %s",
    level=logging.ERROR,
)
# One failure for three causes, because the operator has to look at the
# account either way: a 403 is as often an unavailable model or region as a
# bad key, and a 404 is the account losing access to a model this bot pins.
FAILURE_REFUSED = _Failure(
    ERROR_NEEDS_OPERATOR,
    "OpenAI refused the key or the model, turning away %s",
    level=logging.ERROR,
)
FAILURE_OUTAGE = _Failure(ERROR_OUTAGE, "OpenAI is failing on its side for %s")
FAILURE_TOO_LONG = _Failure(ERROR_TOO_LONG, "Request too long for the model: %s")
FAILURE_UNKNOWN = _Failure(
    ERROR_GENERIC, "Unexpected error for %s", level=logging.ERROR
)

# Prefix that marks a button as ours, so the handler ignores any other
# inline keyboard added later. Full payload: "fb:<verdict>:<intent>".
CALLBACK_PREFIX = "fb"
# Only real answers are worth rating; a thumbs-down on an out-of-scope
# rejection or an error message teaches the classifier nothing.
FEEDBACK_INTENTS = frozenset({INTENT_KNOWLEDGE_QUESTION, INTENT_TRANSLATION})
FEEDBACK_THANKS = "Thanks!"
FEEDBACK_LOST = "Sorry, I can no longer find the question this answered."

# Telegram compresses every photo to JPEG, so a PhotoSize needs no mime check.
PHOTO_MIME_TYPE = "image/jpeg"

# Prefix marking the "read this large file?" buttons, kept apart from the
# feedback prefix so each callback handler only ever sees its own payloads.
DOCUMENT_PREFIX = "doc"
DOCUMENT_READ = "read"
DOCUMENT_CANCEL = "cancel"

LARGE_FILE_QUESTION = (
    "That is a large file ({megabytes:.1f} MB). Sent as a file it is read at "
    "full resolution, which is worth it when the small print matters but uses "
    "several times more of the bot's capacity than an ordinary photo.\n\n"
    "Read it?"
)
LARGE_FILE_CANCELLED = (
    "Not reading it. Send the same page as an ordinary Telegram photo if you "
    "would rather keep it quick."
)
LARGE_FILE_LOST = "Sorry, I can no longer find the file this was about."
# Telegram rejects a message over this outright, with BadRequest rather
# than a truncated send, so a long document reading arrived as no answer
# at all.
TELEGRAM_MESSAGE_LIMIT = 4096

# How many messages /export sends when asked for no particular number.
EXPORT_LIMIT = 20
EXPORT_EMPTY = "There is nothing to export - we have not spoken yet."
EXPORT_CAPTION = "The last {shown} of {total} remembered messages."
RESET_DONE = (
    "Forgotten - I no longer remember what we were talking about. Your "
    "saved preferences are untouched; use /pref clear for those."
)
RESET_EMPTY = "There was nothing to forget - we have not spoken yet."

# Operator commands. They answer only in the admin chat, and stay silent
# everywhere else rather than announcing that they exist.
LOGLEVEL_USAGE = (
    "Sending {current} and above to this chat.\n\n"
    "/loglevel warning - also send warnings\n"
    "/loglevel error - only send errors"
)
LOGLEVEL_SET = "Sending {level} and above to this chat."
LOGS_EMPTY = "Nothing logged since the bot started."
LOGS_CAPTION = "The last {count} log records held in memory."

LIMITS_ALLOWANCES = "Daily allowances: {text} questions, {photo} photos per chat."
LIMITS_SPENT = (
    "Spent today: {text_spent} questions and {photo_spent} photos "
    "across {chats} chats."
)
LIMITS_STATUS = (
    LIMITS_ALLOWANCES + "\n" + LIMITS_SPENT + "\n\n"
    "/limits text <number> - change the question allowance\n"
    "/limits photo <number> - change the photo allowance"
)
LIMITS_BAD_ARGS = (
    "I could not read that as a change. Name the allowance and a whole "
    "number:\n"
    "/limits text 30\n"
    "/limits photo 5"
)
# What an ordinary user sees. Their own figures only - the totals across
# chats are the operator's business, not theirs.
LIMITS_PERSONAL = (
    "Your daily allowance: {text} questions and {photo} photos.\n"
    "Used today: {text_used} questions and {photo_used} photos.\n\n"
    "It resets at midnight UTC."
)

# Prefix marking the "announce this?" buttons, kept apart from the other
# three so each callback handler only ever sees its own payloads.
ANNOUNCEMENT_PREFIX = "ann"
ANNOUNCEMENT_SEND = "send"
ANNOUNCEMENT_SKIP = "skip"
ANNOUNCEMENT_SEND_LABEL = "Tell everyone"
ANNOUNCEMENT_SKIP_LABEL = "Not now"
ANNOUNCEMENT_SENT = "Posted to the announcements channel."
ANNOUNCEMENT_SKIPPED = "Nothing posted."
# One message read by everyone, so it cannot carry a reader's own spend the
# way a mailing to each chat could. It points at the command that can.
ANNOUNCEMENT_LIMITS = (
    "Daily allowances have changed.\n\n"
    "Each chat can now ask {text} questions and send {photo} photos per day. "
    "The count resets at midnight UTC.\n\n"
    "Send /limits to the bot to see how much of yours is left today."
)

# Where users are told the announcements are. Appended to /start and /help
# only when a channel is configured, since an invitation to nowhere is worse
# than no invitation.
CHANNEL_INVITATION = (
    "\n\nNews about the bot - allowance changes, planned downtime, new "
    "features - is posted at {link}. Subscribe there to hear it."
)

# Prefix marking the operator panel's buttons, kept apart from the other
# three so each callback handler only ever sees its own payloads.
ADMIN_PREFIX = "adm"
# The panel repeats what /limits says, and the spent line is load-bearing
# rather than decorative: resetting the counters changes nothing else on the
# panel, so without it Telegram would refuse every redraw as an edit that
# changes nothing and the operator would never see the reset land.
ADMIN_PANEL = (
    "Operator panel\n\n" + LOGLEVEL_SET + "\n" + LIMITS_ALLOWANCES + "\n" + LIMITS_SPENT
)
LEVEL_IN_FORCE = "●"
LEVEL_AVAILABLE = "○"

# The one panel action that is neither a level nor a switch: it does
# something once rather than moving anything into a position.
ADMIN_RESET_COUNTERS = "counters"
RESET_COUNTERS_LABEL = "Reset today's counters"
COUNTERS_RESET = "Today's counters are back to zero."

# The commands the operator can switch off from the panel. Both rest on:
# the switch is there for an incident, so a bot nobody has touched works.
FEATURE_EXPORT = "export"
FEATURE_RESET = "reset"
FEATURES = (FEATURE_EXPORT, FEATURE_RESET)
FEATURE_LABEL = "/{name}: {state}"
FEATURE_ON = "on"
FEATURE_OFF = "off"
FEATURE_UNAVAILABLE = (
    "/{name} is not available at the moment - the bot's operator has "
    "switched it off. Everything else still works."
)


UNSUPPORTED_FILE = (
    "I can't read {kind} files yet. Please send a photo or a screenshot "
    "of the page instead."
)

EDITED_MESSAGE_NOTICE = (
    "I only see a message as it was first sent, so editing one does not "
    "reach me. Send it again as a new message and I'll act on it."
)


@dataclass(frozen=True)
class _PanelState:
    """Everything the operator panel shows, read at the moment it is drawn.

    Attributes:
        push_level: The threshold the log handler is pushing at.
        features: Which switchable commands are on, keyed by name.
        limits_report: The allowances and today's spend, as _limits_report
            returns them.
    """

    push_level: int
    features: dict[str, bool]
    limits_report: dict[str, int]


def _with_channel_invitation(text: str) -> str:
    """Add the announcements channel to a message, when there is one.

    The channel is opt-in - nothing reaches a user who has not subscribed -
    so the two commands that introduce the bot are where its address belongs.

    Args:
        text: The message about to be sent.

    Returns:
        The same message with the invitation appended, or unchanged when no
        channel is configured.
    """
    name = channel_name()
    if name is None:
        return text
    return text + CHANNEL_INVITATION.format(link=name)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /start command. Sends a welcome message with bot capabilities."""
    welcome_text = (
        "Welcome to the Immigrant Assistance Bot!\n\n"
        "I can help you with:\n"
        "- Information about living in Serbia (residency, documents, etc.)\n"
        "- Translation between Serbian and English\n"
        "- Reading a Serbian document - just send me a photo of it\n\n"
        "Just send me a message with your question, or use /help "
        "to see available commands."
    )
    await update.message.reply_text(_with_channel_invitation(welcome_text))


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /help command. Lists available commands and usage examples."""
    help_text = (
        "Available commands:\n\n"
        "/start - Welcome message\n"
        "/help - Show this help message\n"
        "/pref - Set standing preferences (e.g. always reply in Cyrillic)\n"
        "/reset - Forget our conversation and start fresh\n"
        "/export - Send our recent messages back as a text file\n"
        "/limits - See your daily allowance and what is left of it\n\n"
        "You can also just type a question and I'll try to help, or send a "
        "photo of a Serbian document and I'll explain it.\n\n"
        "Examples:\n"
        '- "How do I apply for a temporary residence permit?"\n'
        '- "Translate: Dobro jutro, kako ste?"\n'
        "- A photo of a bill, with or without a caption asking about it"
    )
    await update.message.reply_text(_with_channel_invitation(help_text))


PREF_USAGE = (
    "Preferences are standing rules I apply to your future replies "
    "(mainly how to translate).\n\n"
    "/pref - show your saved preferences\n"
    "/pref add <rule> - save a rule, e.g.\n"
    "    /pref add Write Serbian translations in Cyrillic\n"
    "/pref remove <number> - delete the rule with that number\n"
    "/pref tidy - merge rules that mean the same thing\n"
    "/pref clear - remove all your preferences"
)


def _numbered_list(rules: list[str]) -> str:
    """Render preference rules as a 1-based numbered block for /pref remove."""
    return "\n".join(f"{position}. {rule}" for position, rule in enumerate(rules, 1))


async def _pref_add(update: Update, orchestrator, chat_id, args: list[str]) -> None:
    """Save a new standing rule from `/pref add <rule>`."""
    rule = " ".join(args[1:]).strip()
    if not rule:
        await update.message.reply_text(
            "Tell me the rule to save, e.g.\n"
            "/pref add Write Serbian translations in Cyrillic"
        )
        return
    rules = await add_preference(orchestrator, chat_id, rule)
    saved = "Saved. Your preferences:\n" + _numbered_list(rules)
    await update.message.reply_text(saved)


async def _pref_remove(update: Update, orchestrator, chat_id, args: list[str]) -> None:
    """Delete one rule by its 1-based number from `/pref remove <number>`."""
    rules = await get_preferences(orchestrator, chat_id)
    if len(args) < 2 or not args[1].isdigit():
        await update.message.reply_text("Which one? Use /pref remove <number>.")
        return
    number = int(args[1])
    if number < 1 or number > len(rules):
        await update.message.reply_text(f"There is no preference number {number}.")
        return
    remaining = await remove_preference(orchestrator, chat_id, number - 1)
    if not remaining:
        await update.message.reply_text("Removed. You have no saved preferences now.")
        return
    await update.message.reply_text(
        "Removed. Your preferences:\n" + _numbered_list(remaining)
    )


async def _pref_tidy(update: Update, context, orchestrator, chat_id) -> None:
    """Merge semantically-duplicate rules on demand for `/pref tidy`."""
    tidier = context.bot_data["preference_tidier"]
    # The only /pref branch that can wait on a model, so the only one that
    # shows the indicator, and the only one that can fail the way an agent
    # call fails. The try covers the indicator too, so it stops before the
    # apology is sent.
    try:
        async with show_typing(context.bot, chat_id):
            rules = await tidy_preferences(orchestrator, tidier, chat_id)
    except Exception as error:
        user_id = update.message.from_user.id
        await _reply_with_error(update.message, error, f"user {user_id}: /pref tidy")
        return
    if not rules:
        await update.message.reply_text("You have no saved preferences to tidy.")
        return
    tidied = "Tidied. Your preferences:\n" + _numbered_list(rules)
    await update.message.reply_text(tidied)


async def pref_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /pref command: view, add, remove, tidy, or clear preferences.

    With no arguments it shows the saved rules. Subcommands manage them:
    `add <rule>`, `remove <number>`, `tidy` (merge duplicates), and `clear`.
    Preferences are kept per chat in the orchestrator's checkpointer, so each
    user has their own.
    """
    orchestrator = context.bot_data["orchestrator"]
    chat_id = update.message.chat_id
    args = context.args

    if not args:
        rules = await get_preferences(orchestrator, chat_id)
        if not rules:
            await update.message.reply_text(
                "You have no saved preferences yet.\n\n" + PREF_USAGE
            )
            return
        await update.message.reply_text(
            "Your saved preferences:\n" + _numbered_list(rules)
        )
        return

    action = args[0].lower()
    if action == "add":
        await _pref_add(update, orchestrator, chat_id, args)
        return
    if action == "remove":
        await _pref_remove(update, orchestrator, chat_id, args)
        return
    if action == "tidy":
        await _pref_tidy(update, context, orchestrator, chat_id)
        return
    if action == "clear":
        await clear_preferences(orchestrator, chat_id)
        await update.message.reply_text("Cleared. You have no saved preferences now.")
        return

    await update.message.reply_text(PREF_USAGE)


def _export_limit(args: list) -> int:
    """Read how many messages /export was asked for, defaulting sensibly."""
    if args and args[0].isdigit():
        return int(args[0])
    return EXPORT_LIMIT


async def export_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /export: send this chat's recent messages back as a text file.

    Sent as a file rather than a message because Telegram caps a message at
    4096 characters and a handful of document readings passes that easily,
    and because a file can be forwarded or attached to a bug report whole,
    which is the point of having it at all.

    `/export 50` overrides how many messages to include.
    """
    if not _is_enabled(context, FEATURE_EXPORT):
        await update.message.reply_text(FEATURE_UNAVAILABLE.format(name=FEATURE_EXPORT))
        return

    orchestrator = context.bot_data["orchestrator"]
    chat_id = update.message.chat_id
    messages = await get_history(orchestrator, chat_id)
    if not messages:
        await update.message.reply_text(EXPORT_EMPTY)
        return

    limit = _export_limit(context.args)
    transcript = format_transcript(messages, limit)
    await update.message.reply_document(
        document=BytesIO(transcript.encode("utf-8")),
        filename=f"settlein-chat-{chat_id}.txt",
        caption=EXPORT_CAPTION.format(
            shown=min(limit, len(messages)), total=len(messages)
        ),
    )


async def loglevel_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /loglevel: choose which records are pushed to the admin chat.

    Only the handler's own threshold moves. No logger's level is touched, so
    the server keeps logging everything it logged before, and nothing a
    library writes can be turned on from a chat - which matters because
    httpx logs the bot token as part of every request URL.
    """
    if not is_admin(update.message.chat_id):
        return

    handler = context.bot_data.get("log_handler")
    if handler is None:
        return

    wanted = context.args[0].lower() if context.args else ""
    if wanted not in LOG_LEVELS:
        current = logging.getLevelName(handler.push_level)
        await update.message.reply_text(LOGLEVEL_USAGE.format(current=current))
        return

    handler.push_level = LOG_LEVELS[wanted]
    await update.message.reply_text(LOGLEVEL_SET.format(level=wanted.upper()))


async def logs_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /logs: send the buffered log records back as a file.

    Sent as a file for the same reason /export is: a few hundred lines pass
    Telegram's message cap, and a file can be attached to a bug report
    whole. The buffer lives in memory, so a restart empties it - at which
    point the server's own log is where to look.
    """
    if not is_admin(update.message.chat_id):
        return

    handler = context.bot_data.get("log_handler")
    if handler is None:
        return

    records = handler.snapshot()
    if not records:
        await update.message.reply_text(LOGS_EMPTY)
        return

    await update.message.reply_document(
        document=BytesIO("\n".join(records).encode("utf-8")),
        filename="settlein-log.txt",
        caption=LOGS_CAPTION.format(count=len(records)),
    )


async def limits_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /limits: show a chat its allowance, or let the operator move it.

    Unlike /loglevel and /logs this answers everyone, because the allowance
    is the user's own business: someone turned away at their limit has to be
    able to see what the limit is and how much of it is left. Only the
    operator can move it, and only they see the totals across chats.

    The numbers are a guess until the bot has been used, so they are moved
    from a chat rather than only from the environment. A restart puts them
    back to DAILY_TEXT_LIMIT and DAILY_PHOTO_LIMIT, so a number worth keeping
    belongs in .env.
    """
    quota = context.bot_data["daily_quota"]
    chat_id = update.message.chat_id
    if not is_admin(chat_id):
        await update.message.reply_text(
            LIMITS_PERSONAL.format(**_personal_report(quota, chat_id))
        )
        return

    await _run_operator_limits(update.message, context, quota)


async def _run_operator_limits(
    message, context: ContextTypes.DEFAULT_TYPE, quota: DailyQuota
) -> None:
    """Show the operator's view of the allowances, or move one of them.

    Args:
        message: The operator's message, to reply to.
        context: The PTB context, carrying the command's arguments.
        quota: The quota being read or changed.
    """
    if not context.args:
        await message.reply_text(_limits_status(quota))
        return

    change = _parse_limit_change(context.args)
    if change is None:
        # Said rather than ignored: replying with the unchanged status would
        # look identical to a change that landed, and a typo in a number is
        # exactly the mistake this command invites.
        await message.reply_text(LIMITS_BAD_ARGS)
        return

    kind, number = change
    quota.limits[kind] = number
    # Offered rather than posted outright: several numbers get tried in a
    # row, and announcing each one would tell subscribers about allowances
    # the operator has already moved past.
    await message.reply_text(
        _limits_status(quota), reply_markup=_announcement_keyboard()
    )


def _parse_limit_change(args: list[str]) -> tuple[str, int] | None:
    """Read which allowance /limits was asked to move, and where to.

    Args:
        args: The words after the command.

    Returns:
        The kind and its new limit, or None when the command was not a
        well-formed change - including a bare /limits asking only to look.
    """
    if len(args) != 2:
        return None
    kind = args[0].lower()
    # isdigit() also turns away a negative, and lets 0 through: switching a
    # kind off entirely is a real thing to want during an incident.
    if kind not in QUOTA_KINDS or not args[1].isdigit():
        return None
    return kind, int(args[1])


def _limits_report(quota: DailyQuota) -> dict[str, int]:
    """Gather the numbers both /limits and the operator panel report.

    One reader for both, so the panel cannot drift into saying something
    different from the command about the same allowances.

    Args:
        quota: The DailyQuota holding the limits and the counters.

    Returns:
        The fields LIMITS_ALLOWANCES and LIMITS_SPENT are formatted with.
    """
    spent = quota.spent_today()
    return {
        "text": quota.limits[KIND_TEXT],
        "photo": quota.limits[KIND_PHOTO],
        "text_spent": spent[KIND_TEXT],
        "photo_spent": spent[KIND_PHOTO],
        "chats": quota.active_chats(),
    }


def _personal_report(quota: DailyQuota, chat_id: int) -> dict[str, int]:
    """Gather what one chat is told about its own allowance.

    The companion to _limits_report: same allowances, but this chat's spend
    rather than everyone's, since what one user has spent is their own
    business and the totals across chats are the operator's.

    Args:
        quota: The DailyQuota holding the limits and the counters.
        chat_id: The chat being told.

    Returns:
        The fields LIMITS_PERSONAL is formatted with.
    """
    used = quota.usage(chat_id)
    return {
        "text": quota.limits[KIND_TEXT],
        "photo": quota.limits[KIND_PHOTO],
        "text_used": used[KIND_TEXT],
        "photo_used": used[KIND_PHOTO],
    }


def _limits_status(quota: DailyQuota) -> str:
    """Render both allowances and what today has actually cost so far.

    Args:
        quota: The DailyQuota holding the limits and the counters.

    Returns:
        The message /limits replies with.
    """
    return LIMITS_STATUS.format(**_limits_report(quota))


async def announcement_callback(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Publish a change to the allowances, or drop the offer to.

    Stateless like the other three keyboards, but for a different reason.
    The panel's buttons carry the position a tap puts a switch in, so a tap
    on a stale panel lands where the button said. This one carries nothing
    and reads the allowances at the moment it is tapped: a button that sets
    something should honour what it promised, while a button that publishes
    something must not announce a number already moved past.
    """
    query = update.callback_query
    action = _parse_announcement_action(query.data)
    # Its own guard, not the "/" menu's scoping - that is presentation only.
    if not _is_admin_tap(query) or action is None:
        await query.answer()
        return

    # Clearing the buttons doubles as a lock, the way it does for a large
    # file: Telegram refuses the edit once they are gone, which is the only
    # sign a second tap got here first, and the same news must not be posted
    # to the channel twice.
    if not await _hide_keyboard(query):
        await query.answer()
        return

    if action == ANNOUNCEMENT_SKIP:
        await query.answer(ANNOUNCEMENT_SKIPPED)
        return

    quota = context.bot_data["daily_quota"]
    reason = await post_announcement(
        context.bot, ANNOUNCEMENT_LIMITS.format(**_limits_report(quota))
    )
    if reason is None:
        await query.answer(ANNOUNCEMENT_SENT)
        return

    # Said in the chat rather than as a toast: a toast is easy to miss, and
    # an announcement that did not go out is the one case here the operator
    # has to do something about.
    await query.answer()
    await query.message.reply_text(reason)


def _announcement_keyboard() -> InlineKeyboardMarkup | None:
    """Offer to tell subscribers what just changed.

    Returns:
        The two buttons, or None when no channel is configured - an offer to
        publish somewhere that does not exist is worse than no offer.
    """
    if channel_name() is None:
        return None
    buttons = [
        InlineKeyboardButton(
            ANNOUNCEMENT_SEND_LABEL,
            callback_data=f"{ANNOUNCEMENT_PREFIX}:{ANNOUNCEMENT_SEND}",
        ),
        InlineKeyboardButton(
            ANNOUNCEMENT_SKIP_LABEL,
            callback_data=f"{ANNOUNCEMENT_PREFIX}:{ANNOUNCEMENT_SKIP}",
        ),
    ]
    return InlineKeyboardMarkup([buttons])


def _parse_announcement_action(callback_data: str) -> str | None:
    """Read which of the two announcement buttons was tapped.

    Buttons outlive deploys: an offer sent before this format existed is
    still tappable in the chat history, so an unknown payload is rejected
    rather than acted on - and acting on this one publishes to a channel.

    Args:
        callback_data: The raw payload from the tapped button.

    Returns:
        ANNOUNCEMENT_SEND, ANNOUNCEMENT_SKIP, or None when the payload is not
        one this keyboard issues.
    """
    prefix, _, action = callback_data.partition(":")
    if prefix != ANNOUNCEMENT_PREFIX:
        return None
    if action not in (ANNOUNCEMENT_SEND, ANNOUNCEMENT_SKIP):
        return None
    return action


async def admin_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /admin: show the panel the operator runs the bot from.

    A control surface, not a log viewer: it says what the bot is doing now
    and offers the taps that change it. Records still arrive by push (the
    mirror) and by pull (/logs).
    """
    if not is_admin(update.message.chat_id):
        return

    handler = context.bot_data.get("log_handler")
    # Silent for the same reason a non-admin gets silence: without the
    # mirror there is no state to show, and nothing the operator taps could
    # do anything. Only reachable if the mirror failed to install.
    if handler is None:
        return

    state = _panel_state(context)
    await update.message.reply_text(
        _admin_panel_text(state), reply_markup=_admin_keyboard(state)
    )


async def admin_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Apply a tap on the operator panel, then redraw the panel in place.

    Stateless like the other two keyboards: what to change travels in the
    button's payload, and the state it changes is read back out of bot_data,
    so nothing is held between sending the panel and a tap on it.
    """
    query = update.callback_query
    handler = context.bot_data.get("log_handler")
    action = _parse_admin_action(query.data)
    if not _is_admin_tap(query) or handler is None or action is None:
        await query.answer()
        return

    changed = _apply_admin_action(context, action)
    if await _redraw_admin_panel(query, _panel_state(context)):
        await query.answer()
        return
    # The panel could not be rewritten, so the toast carries what it would
    # have said - the tap changed something either way.
    await query.answer(changed)


def _is_admin_tap(query: CallbackQuery) -> bool:
    """Say whether a tap came from the operator's own chat.

    Reads the chat off `message.chat` rather than the `chat_id` shortcut:
    past about 48 hours Telegram sends an InaccessibleMessage, which carries
    a chat but none of the shortcuts Message defines on top of it.

    Args:
        query: The CallbackQuery raised by the button tap.

    Returns:
        True if the panel tapped is the one in the admin chat.
    """
    if query.message is None:
        return False
    return is_admin(query.message.chat.id)


def _apply_admin_action(context: ContextTypes.DEFAULT_TYPE, action: str) -> str:
    """Carry out one panel tap, and describe the state it leaves behind.

    Args:
        context: The PTB context, holding the panel's state in bot_data.
        action: The action the tapped button carried.

    Returns:
        A line naming the new state, which the toast falls back on when the
        panel itself cannot be redrawn.
    """
    if action in LOG_LEVELS:
        context.bot_data["log_handler"].push_level = LOG_LEVELS[action]
        return LOGLEVEL_SET.format(level=action.upper())

    if action == ADMIN_RESET_COUNTERS:
        context.bot_data["daily_quota"].reset()
        return COUNTERS_RESET

    return _set_feature(context, action)


def _panel_state(context: ContextTypes.DEFAULT_TYPE) -> _PanelState:
    """Gather what the panel shows, at the moment it is about to be drawn.

    Args:
        context: The PTB context; its bot_data holds the log handler the
            caller has already found.

    Returns:
        The level being pushed, the position of every switch, and the
        allowances in force.
    """
    return _PanelState(
        push_level=context.bot_data["log_handler"].push_level,
        features=_feature_states(context),
        limits_report=_limits_report(context.bot_data["daily_quota"]),
    )


def _admin_panel_text(state: _PanelState) -> str:
    """Say what the panel reports in words rather than on its buttons.

    Args:
        state: What the panel is showing.

    Returns:
        The panel's message body.
    """
    return ADMIN_PANEL.format(
        level=logging.getLevelName(state.push_level), **state.limits_report
    )


def _admin_keyboard(state: _PanelState) -> InlineKeyboardMarkup:
    """Build the panel's buttons: the level to push at, then the switches.

    Args:
        state: What the panel is showing.

    Returns:
        Three rows: the push levels, the switches - each labelled with the
        state it is in now and carrying the action a tap on it performs -
        and the one button that does something rather than setting
        something.
    """
    levels = [
        _admin_button(_level_label(name, state.push_level), name) for name in LOG_LEVELS
    ]
    switches = [_feature_button(name, state.features) for name in FEATURES]
    counters = [_admin_button(RESET_COUNTERS_LABEL, ADMIN_RESET_COUNTERS)]
    return InlineKeyboardMarkup([levels, switches, counters])


def _feature_button(name: str, features: dict[str, bool]) -> InlineKeyboardButton:
    """Build one switch button, which reads and acts on different things.

    The label is the position the switch is in now; the payload is the
    position a tap puts it in. Naming the target rather than asking for a
    flip is what makes a tap on a panel drawn before someone else moved the
    same switch land where the button said it would.

    Args:
        name: The command the switch controls.
        features: Which switchable commands are on.

    Returns:
        The button, ready to place in a row.
    """
    target = FEATURE_OFF if features[name] else FEATURE_ON
    return _admin_button(_feature_label(name, features), f"{name}:{target}")


def _admin_button(label: str, action: str) -> InlineKeyboardButton:
    """Build one panel button, tagged with the prefix that routes its tap.

    Args:
        label: The text shown on the button.
        action: What tapping it asks the panel to do.

    Returns:
        The button, ready to place in a row.
    """
    return InlineKeyboardButton(label, callback_data=f"{ADMIN_PREFIX}:{action}")


def _level_label(name: str, push_level: int) -> str:
    """Name one level button, marked when it is the level in force.

    Args:
        name: The level this button switches to.
        push_level: The threshold the log handler is pushing at.

    Returns:
        The text shown on the button.
    """
    marker = LEVEL_IN_FORCE if LOG_LEVELS[name] == push_level else LEVEL_AVAILABLE
    return f"{marker} {name.upper()}"


def _feature_label(name: str, features: dict[str, bool]) -> str:
    """Name one switch button, showing the position it is in now.

    Args:
        name: The command the switch controls.
        features: Which switchable commands are on.

    Returns:
        The text shown on the button.
    """
    state = FEATURE_ON if features[name] else FEATURE_OFF
    return FEATURE_LABEL.format(name=name, state=state)


def default_features() -> dict[str, bool]:
    """Return every switch in the position the bot starts in.

    Read at startup to seed bot_data, and again wherever a flag is missing,
    so the shipped position is written down once rather than in each place
    that has to cope with its absence.

    Returns:
        One entry per switchable command.
    """
    return dict.fromkeys(FEATURES, True)


def _is_enabled(context: ContextTypes.DEFAULT_TYPE, feature: str) -> bool:
    """Say whether a command the operator can switch off is on right now.

    A missing flag reads as whatever the bot ships with. The switches are an
    incident control, so the bot has to work in a run where nobody has
    touched them - including one where the panel was never wired up at all.

    Args:
        context: The PTB context, holding the flags in bot_data.
        feature: The command being asked about.

    Returns:
        True if the command may run.
    """
    return (default_features() | context.bot_data.get("features", {}))[feature]


def _feature_states(context: ContextTypes.DEFAULT_TYPE) -> dict[str, bool]:
    """Read the position of every switch, for the panel to draw.

    Args:
        context: The PTB context, holding the flags in bot_data.

    Returns:
        One entry per switchable command.
    """
    return {name: _is_enabled(context, name) for name in FEATURES}


def _set_feature(context: ContextTypes.DEFAULT_TYPE, action: str) -> str:
    """Put one switch where its button said, so its command stops or starts.

    Args:
        context: The PTB context, holding the flags in bot_data.
        action: The tapped payload, as "<command>:<position>".

    Returns:
        A line naming the position the switch is now in.
    """
    name, _, position = action.partition(":")
    features = context.bot_data.setdefault("features", {})
    features[name] = position == FEATURE_ON
    return _feature_label(name, _feature_states(context))


def _parse_admin_action(callback_data: str) -> str | None:
    """Read which of the panel's buttons was tapped.

    Buttons outlive deploys: a panel sent before this format existed is
    still tappable in the chat history, so unknown payloads are rejected
    rather than acted on.

    Args:
        callback_data: The raw payload from the tapped button.

    Returns:
        The action to carry out - a log level, the counter reset, or a
        switchable command and the position to put it in - or None if the
        payload is not one this panel issues.
    """
    parts = callback_data.split(":", maxsplit=1)
    if len(parts) != 2 or parts[0] != ADMIN_PREFIX:
        return None
    action = parts[1]
    if action in LOG_LEVELS or action == ADMIN_RESET_COUNTERS:
        return action
    name, _, position = action.partition(":")
    if name in FEATURES and position in (FEATURE_ON, FEATURE_OFF):
        return action
    return None


async def _redraw_admin_panel(query: CallbackQuery, state: _PanelState) -> bool:
    """Rewrite the panel to show the state the tap has just left it in.

    The whole message is rebuilt from bot_data rather than the tapped button
    being patched, so a panel left open while /loglevel changed the level
    redraws correct rather than merely different.

    Args:
        query: The CallbackQuery raised by the button tap.
        state: What the panel should now be showing.

    Returns:
        True if the panel now shows that state. False if it could not be
        rewritten - Telegram refuses an edit that would change nothing, one
        whose message has been deleted, and one past about 48 hours, which
        python-telegram-bot raises as TypeError rather than BadRequest. The
        tap has already changed something in every one of those cases, so
        the caller has to say so some other way.
    """
    try:
        await query.edit_message_text(
            _admin_panel_text(state), reply_markup=_admin_keyboard(state)
        )
    except (BadRequest, TypeError):
        logger.debug("Operator panel could not be redrawn (unchanged or stale).")
        return False
    return True


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /reset: forget this chat's conversation, keeping its preferences.

    Memory is per chat and otherwise unbounded until it hits
    MAX_STORED_MESSAGES, so without this there is no way to start a new topic
    without the old one colouring the answers - and no way to retest a
    failure honestly, since a correction stays in context and the model
    answers from it rather than working the problem again.
    """
    if not _is_enabled(context, FEATURE_RESET):
        await update.message.reply_text(FEATURE_UNAVAILABLE.format(name=FEATURE_RESET))
        return

    orchestrator = context.bot_data["orchestrator"]
    forgotten = await clear_history(orchestrator, update.message.chat_id)
    await update.message.reply_text(RESET_DONE if forgotten else RESET_EMPTY)


def _feedback_keyboard(intent: str) -> InlineKeyboardMarkup | None:
    """Build the thumbs up/down keyboard for an answer worth rating.

    The intent travels inside the button's callback data, so the tap can be
    recorded later without storing anything server-side.

    Args:
        intent: The intent the orchestrator settled on for this reply.

    Returns:
        A two-button keyboard, or None for intents we do not collect
        feedback on (out-of-scope rejections).
    """
    if intent not in FEEDBACK_INTENTS:
        return None
    buttons = [
        InlineKeyboardButton(
            "👍", callback_data=f"{CALLBACK_PREFIX}:{VERDICT_UP}:{intent}"
        ),
        InlineKeyboardButton(
            "👎", callback_data=f"{CALLBACK_PREFIX}:{VERDICT_DOWN}:{intent}"
        ),
    ]
    return InlineKeyboardMarkup([buttons])


# Markdown the models keep emitting despite being told not to. Telegram is
# sent plain text (no parse_mode), so these arrive as literal characters.
_MARKDOWN_HEADING = re.compile(r"^#{1,6}[ \t]+", flags=re.MULTILINE)
_MARKDOWN_EMPHASIS = re.compile(r"(\*\*|__)(.+?)\1", flags=re.DOTALL)


def _strip_markdown(text: str) -> str:
    """Remove the markdown Telegram would show as raw characters.

    Replies are sent without a parse_mode on purpose: MarkdownV2 demands that
    `.`, `-` and `(` be escaped, and an answer quoting "2.066,98" or a phone
    number would be rejected outright, leaving the user with an error instead
    of an answer. Cleaning the text cannot fail that way.

    Only headings and paired bold markers are touched. A lone `*` or `_` is
    left alone, since it is likelier to be part of the document being quoted
    than a formatting attempt.

    Args:
        text: The model's answer.

    Returns:
        The same answer with `#` headings and `**bold**` markers removed.
    """
    without_headings = _MARKDOWN_HEADING.sub("", text)
    return _MARKDOWN_EMPHASIS.sub(r"\2", without_headings)


def _split_for_telegram(text: str) -> list[str]:
    """Break an answer into pieces Telegram will accept.

    Splits at a line break inside the last window where one exists, so a
    numbered list or a table row is not cut in half. Falls back to a hard cut
    only when a single line is longer than the whole limit.

    Args:
        text: The answer to send.

    Returns:
        One or more pieces, each within TELEGRAM_MESSAGE_LIMIT.
    """
    parts = []
    remaining = text
    while len(remaining) > TELEGRAM_MESSAGE_LIMIT:
        window = remaining[:TELEGRAM_MESSAGE_LIMIT]
        cut = window.rfind("\n")
        if cut <= 0:
            cut = TELEGRAM_MESSAGE_LIMIT
        parts.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip("\n")
    parts.append(remaining)
    return [part for part in parts if part] or [text]


def _classify_openai_failure(error: Exception) -> _Failure:
    """Decide which kind of failure an exception represents.

    Order matters twice over. APITimeoutError subclasses APIConnectionError,
    so it has to be tested first or it becomes unreachable. And no
    APIStatusError or APIError catch-all may be added above the branches
    below, because every one of them is a subclass of it.

    Args:
        error: The exception the AI call raised.

    Returns:
        What to tell the user, what to log, and whom to wake.
    """
    if isinstance(error, APITimeoutError):
        return FAILURE_TIMEOUT
    if isinstance(error, APIConnectionError):
        return FAILURE_CONNECTION
    if isinstance(error, RateLimitError):
        # Both arrive as 429. `code` is None when the response carried no
        # JSON body, which correctly falls through to plain throttling.
        if error.code == QUOTA_EXHAUSTED_CODE:
            return FAILURE_QUOTA
        return FAILURE_THROTTLED
    # 404 belongs here: the only one this bot can provoke is a model its
    # account has lost access to, which no user can do anything about.
    if isinstance(error, (AuthenticationError, PermissionDeniedError, NotFoundError)):
        return FAILURE_REFUSED
    if isinstance(error, InternalServerError):
        return FAILURE_OUTAGE
    # One condition, not a nested `if`: every other 400 has to fall through
    # to FAILURE_UNKNOWN and be logged with its traceback.
    if isinstance(error, BadRequestError) and error.code == CONTEXT_LENGTH_CODE:
        return FAILURE_TOO_LONG
    return FAILURE_UNKNOWN


async def _reply_with_error(message, error: Exception, request: str) -> None:
    """Log a failed AI call and send the user the matching apology.

    Shared by the text and the document handler so that a timeout looks the
    same to the user whichever one they hit. The user never sees a stack
    trace; the detail stays in the server log.

    Args:
        message: The user's message, to reply to.
        error: The exception the AI call raised.
        request: Short description of what was being processed, for the log.
    """
    failure = _classify_openai_failure(error)
    logger.log(
        failure.level,
        failure.log,
        request,
        exc_info=error if failure is FAILURE_UNKNOWN else None,
    )
    await message.reply_text(failure.reply)


def _check_daily_quota(
    context: ContextTypes.DEFAULT_TYPE, chat_id: int, kind: str
) -> None:
    """Refuse a message this chat no longer has the allowance for.

    Spends nothing; `_record_daily_message` does that once the answer has
    been produced. The operator's own chat is exempt from both, so an
    incident cannot leave the person fixing it locked out of the bot, and a
    demo cannot throttle the person giving it. The exemption lives here
    rather than inside DailyQuota to keep middleware.py free of any import
    from the rest of src/.

    Args:
        context: The PTB context, holding the quota in bot_data.
        chat_id: The chat the message arrived in.
        kind: KIND_TEXT or KIND_PHOTO.

    Raises:
        ValidationError: If this chat has spent that kind's allowance.
    """
    if is_admin(chat_id):
        return
    context.bot_data["daily_quota"].check(chat_id, kind)


def _record_daily_message(
    context: ContextTypes.DEFAULT_TYPE, chat_id: int, kind: str
) -> None:
    """Charge one answered message to this chat's daily allowance.

    Args:
        context: The PTB context, holding the quota in bot_data.
        chat_id: The chat the message arrived in.
        kind: KIND_TEXT or KIND_PHOTO.
    """
    if is_admin(chat_id):
        return
    context.bot_data["daily_quota"].record(chat_id, kind)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle any non-command text message.

    Validates input, checks both limits, then routes through the
    orchestrator. The two limits answer different questions: the rate limiter
    caps how fast one user spends, the daily quota how much this chat spends
    in a day.
    """
    try:
        user_text = validate_message_text(update.message.text)
        rate_limiter = context.bot_data["rate_limiter"]
        rate_limiter.check(update.message.from_user.id)
        _check_daily_quota(context, update.message.chat_id, KIND_TEXT)
    except ValidationError as e:
        await update.message.reply_text(e.user_message)
        return

    orchestrator = context.bot_data["orchestrator"]

    user_id = update.message.from_user.id

    try:
        async with show_typing(context.bot, update.message.chat_id):
            result = await process_message(
                orchestrator, user_text, thread_id=update.message.chat_id
            )
        # Charged here rather than above: the allowance pays for answers, so
        # a question the model never answered does not come out of it.
        _record_daily_message(context, update.message.chat_id, KIND_TEXT)
        # Answering as a Telegram reply is load-bearing, not cosmetic: it is
        # how a later button tap finds the question this answered.
        parts = _split_for_telegram(_strip_markdown(result.response))
        for position, part in enumerate(parts, 1):
            await update.message.reply_text(
                part,
                reply_to_message_id=update.message.message_id,
                # If the question was deleted while the LLM was thinking, send
                # the answer anyway rather than losing a reply we paid for.
                allow_sending_without_reply=True,
                # The buttons go on the last piece: that is where the answer
                # ends, and it is the piece a later tap walks back from.
                reply_markup=(
                    _feedback_keyboard(result.intent)
                    if position == len(parts)
                    else None
                ),
            )
    except Exception as error:
        await _reply_with_error(update.message, error, f"user {user_id}: {user_text}")


def _uploaded_image(message) -> tuple:
    """Pick the image out of a photo message or an image sent as a file.

    Args:
        message: The incoming Telegram message.

    Returns:
        The file to download and its media type.
    """
    if message.photo:
        # Telegram attaches the same photo at several sizes; the last is the
        # largest, and Telegram has already re-encoded all of them as JPEG.
        return message.photo[-1], PHOTO_MIME_TYPE
    return message.document, message.document.mime_type


async def _fetch_document_image(message) -> str:
    """Validate an uploaded image and download it as a data URL.

    Size and type come from the update itself, so an oversized file is
    refused before a byte of it is fetched.

    Args:
        message: The incoming Telegram message.

    Returns:
        The image encoded as a data URL, ready for the vision agent.

    Raises:
        ValidationError: If the image is too large or of an unusable type.
    """
    upload, mime_type = _uploaded_image(message)
    validate_image_upload(upload.file_size, mime_type)
    # Logged because the model is billed by the pixels it is handed, and
    # "what did it actually see" was guesswork until this line existed.
    logger.info(
        "Reading a %s upload of %s bytes (%s)",
        "photo" if message.photo else "file",
        upload.file_size,
        mime_type,
    )
    telegram_file = await upload.get_file()
    image_bytes = await telegram_file.download_as_bytearray()
    return encode_image_as_data_url(bytes(image_bytes), mime_type)


def _confirmation_keyboard() -> InlineKeyboardMarkup:
    """Build the yes/no buttons offered before reading a large file."""
    buttons = [
        InlineKeyboardButton(
            "Read it", callback_data=f"{DOCUMENT_PREFIX}:{DOCUMENT_READ}"
        ),
        InlineKeyboardButton(
            "Cancel", callback_data=f"{DOCUMENT_PREFIX}:{DOCUMENT_CANCEL}"
        ),
    ]
    return InlineKeyboardMarkup([buttons])


async def _read_photo_and_reply(message, context) -> None:
    """Answer the photo on `message` by running it through the orchestrator.

    Everything past the download belongs to the graph: the document agent is
    a node like any other, reached by the entry edge that reads the modality.
    The handler's job is the part only Telegram knows about - checking the
    upload and fetching its bytes.

    Args:
        message: The Telegram message carrying the photo or image file.
        context: The PTB context, holding the orchestrator in bot_data.
    """
    context.bot_data["rate_limiter"].check(message.from_user.id)
    # Checked here because this is the one point every read passes through:
    # a large file confirmed with the buttons arrives by a different route.
    _check_daily_quota(context, message.chat_id, KIND_PHOTO)
    async with show_typing(context.bot, message.chat_id):
        image_url = await _fetch_document_image(message)
        result = await process_document(
            context.bot_data["orchestrator"],
            DocumentTurn(image_url=image_url, caption=message.caption or ""),
            message.chat_id,
        )
    # Charged only now. A photo allowance is small enough that paying for
    # failed downloads and model outages would burn a whole day of it on
    # readings the user never received.
    _record_daily_message(context, message.chat_id, KIND_PHOTO)
    for part in _split_for_telegram(_strip_markdown(result.response)):
        await message.reply_text(part)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Read any photo the user sends and answer their question about it.

    An image carries no text to classify, so modality alone chooses the agent.
    That choice is made by the orchestrator's entry edge, not here: Telegram
    states the modality in the update, so it is settled for free and before
    any model is asked anything.

    A large file is not read straight away. Telegram compresses anything sent
    as a photo, so only an uncompressed upload can be big enough to matter,
    and that one is offered as a choice first.
    """
    message = update.message
    user_id = message.from_user.id
    try:
        upload, mime_type = _uploaded_image(message)
        # Checked before the offer: agreeing to read a file we would then
        # refuse wastes the user's time and makes the question look pointless.
        validate_image_upload(upload.file_size, mime_type)
        # Checked before the offer as well as inside the read. Tapping "Read
        # it" clears the keyboard, so a refusal after the tap would take the
        # offer away with it and leave the sender having to upload again.
        _check_daily_quota(context, message.chat_id, KIND_PHOTO)
        if message.document and is_large_upload(upload.file_size):
            await message.reply_text(
                LARGE_FILE_QUESTION.format(megabytes=upload.file_size / (1024 * 1024)),
                reply_to_message_id=message.message_id,
                allow_sending_without_reply=True,
                reply_markup=_confirmation_keyboard(),
            )
            return
        await _read_photo_and_reply(message, context)
    except ValidationError as e:
        await message.reply_text(e.user_message)
    except Exception as error:
        # Downloading can fail too, not just the model call, so the net is
        # cast around the whole thing rather than the vision call alone.
        await _reply_with_error(message, error, f"user {user_id}: a document photo")


async def document_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Read a large file, or drop it, according to which button was tapped.

    Stateless in the same way as the feedback buttons: nothing is held
    between the question and the answer, because the confirmation was sent as
    a reply to the upload, so the file is found again through
    `query.message.reply_to_message`.
    """
    query = update.callback_query
    # Clearing the buttons doubles as a lock. Telegram refuses the edit when
    # they are already gone, which means another tap got here first, and
    # reading again would pay for the same file twice.
    if not await _hide_keyboard(query):
        await query.answer()
        return

    if query.data == f"{DOCUMENT_PREFIX}:{DOCUMENT_CANCEL}":
        # Answered as a toast rather than a message: the upload may be too
        # old for Telegram to let us reply to it, and there is nothing to say
        # that is worth a second failure path.
        await query.answer(LARGE_FILE_CANCELLED)
        return

    upload = _replied_message(query)
    if upload is None:
        await query.answer(LARGE_FILE_LOST)
        return

    await query.answer()
    try:
        await _read_photo_and_reply(upload, context)
    except ValidationError as e:
        await upload.reply_text(e.user_message)
    except Exception as error:
        user_id = upload.from_user.id
        await _reply_with_error(upload, error, f"user {user_id}: a confirmed file")


def _file_kind(document) -> str:
    """Name the file type in words the sender will recognise.

    Args:
        document: The Telegram Document that arrived.

    Returns:
        An extension such as "DOCX", or a fallback when there is no name.
    """
    name = document.file_name or ""
    suffix = Path(name).suffix.lstrip(".")
    return suffix.upper() if suffix else "that kind of"


async def handle_unsupported_file(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Say which file type arrived and cannot be read, rather than going quiet.

    Registered after the image handler, so it only sees what that one turned
    down: PDFs, documents, archives. A vision model reads images, and
    rendering anything else to one needs tools the container does not carry.
    """
    await update.message.reply_text(
        UNSUPPORTED_FILE.format(kind=_file_kind(update.message.document))
    )


async def handle_edited_message(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Say that an edit was noticed and not acted on, rather than going quiet.

    Telegram delivers an edit as an update of its own, carrying
    `edited_message` and leaving `message` as None - and python-telegram-bot
    matches handlers against `effective_message`, which is the edit. Every
    handler here reads `update.message`, so without this one taking the edit
    first they all run with nothing to read.

    Acting on the edit instead was the alternative, and it is the worse one:
    the up arrow on Telegram Desktop reopens the last message, so a
    half-typed correction would re-run a command, and an edited question
    would be answered a second time and charged to the day's allowance again.
    """
    await update.edited_message.reply_text(EDITED_MESSAGE_NOTICE)


def _parse_feedback(callback_data: str) -> tuple[str, str] | None:
    """Split a button payload into its verdict and intent.

    Buttons outlive deploys: a keyboard sent before this format existed is
    still tappable in the chat history, so unknown payloads are rejected
    rather than unpacked.

    Args:
        callback_data: The raw payload from the tapped button.

    Returns:
        A (verdict, intent) pair, or None if the payload is not ours.
    """
    parts = callback_data.split(":", maxsplit=2)
    if len(parts) != 3 or parts[0] != CALLBACK_PREFIX:
        return None
    _, verdict, intent = parts
    # Both fields are checked against the values we issue, so the stored
    # dataset keeps a closed set of labels to retrain the classifier on.
    if verdict not in (VERDICT_UP, VERDICT_DOWN):
        return None
    if intent not in FEEDBACK_INTENTS:
        return None
    return verdict, intent


def _replied_message(query: CallbackQuery):
    """Recover the message a tapped button's message was replying to.

    Both button features are stateless and rely on this one link: the bot
    answers as a Telegram reply, so a later tap can walk back to whatever
    prompted it without anything being stored in between.

    Args:
        query: The CallbackQuery raised by the button tap.

    Returns:
        The message that was replied to, or None if the chain back to it is
        broken (the user deleted it, or the message is too old for Telegram
        to still hand us its contents).
    """
    tapped = query.message
    # Past about 48 hours Telegram sends an InaccessibleMessage, which
    # carries only a chat and an id - reading anything else raises.
    if tapped is None or not tapped.is_accessible:
        return None
    return tapped.reply_to_message


def _rated_question(query: CallbackQuery) -> str | None:
    """Recover the text of the question that the rated answer replied to.

    Args:
        query: The CallbackQuery raised by the button tap.

    Returns:
        The original question text, or None if it can no longer be reached.
    """
    question = _replied_message(query)
    if question is None:
        return None
    return question.text or None


async def _hide_keyboard(query: CallbackQuery) -> bool:
    """Remove the buttons after a tap, tolerating a message that cannot change.

    Shared by the feedback, large-file and announcement buttons: once a
    choice is made, its buttons go. The result also says whether this call
    was the one that removed them, which is how a second tap on the same
    keyboard is recognised.

    Telegram rejects the edit when the keyboard is already gone, and
    python-telegram-bot raises outright when the message is inaccessible.
    The vote is stored by this point, so neither may reach the user.

    Args:
        query: The CallbackQuery raised by the button tap.

    Returns:
        True if this call removed the buttons, False if they were already
        gone or the message can no longer be edited.
    """
    try:
        await query.edit_message_reply_markup(None)
    except (BadRequest, TypeError):
        logger.debug("Keyboard could not be cleared (stale message).")
        return False
    return True


async def feedback_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Record a thumbs up/down tap on an answer, then clear the buttons.

    Holds no state between the answer and the tap: the question is read
    back from the message the answer replied to.
    """
    query = update.callback_query
    parsed = _parse_feedback(query.data)
    if parsed is None:
        await query.answer()
        return

    question = _rated_question(query)
    if question is None:
        await query.answer(FEEDBACK_LOST)
        return

    verdict, intent = parsed
    try:
        record_feedback(question, intent, verdict)
    except OSError:
        # Feedback is training data, not the user's business: a full or
        # read-only disk must not turn their tap into an error.
        logger.exception("Could not save feedback for intent %s", intent)

    await query.answer(FEEDBACK_THANKS)
    await _hide_keyboard(query)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log an exception no handler caught, and apologise where possible.

    Registered with `add_error_handler`, so it is the last line of defence:
    without it an unexpected crash leaves the user with silence. `update` is
    typed `object` because the failure need not come from a message at all,
    in which case there is nobody to reply to.

    It runs the same failure mapping as the handlers that catch their own
    exceptions. Every path that can raise an OpenAI error already catches
    one, so what arrives here is an error that escaped such a block - and
    the traceback is then the only record of where it came from, which is
    why it is logged here whatever the mapping makes of it.

    Args:
        update: The update being processed when the error happened, if any.
        context: The PTB context, carrying the exception on `.error`.
    """
    logger.error("Unhandled error while processing an update", exc_info=context.error)

    message = getattr(update, "effective_message", None)
    if message is None:
        return
    try:
        await _reply_with_error(message, context.error, f"chat {message.chat_id}")
    except TelegramError:
        # The chat may be blocked or gone; the log above is what matters.
        logger.debug("Could not deliver the error notice to the user.")

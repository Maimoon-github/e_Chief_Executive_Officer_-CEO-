# """
# tools/email_tool.py
# ───────────────────
# Async email sending via aiosmtplib (SMTP).
# Used by Customer Captain (support replies), Promo General (campaigns),
# and the error recovery node (critical alerts).
# """
# from __future__ import annotations

# import json
# import logging
# from email.mime.multipart import MIMEMultipart
# from email.mime.text import MIMEText
# from typing import List, Optional

# import aiosmtplib
# from langchain.tools import BaseTool

# from config.settings import settings

# logger = logging.getLogger(__name__)


# async def send_email(
#     to: str | List[str],
#     subject: str,
#     body_html: str,
#     body_text: Optional[str] = None,
#     reply_to: Optional[str] = None,
# ) -> Dict:
#     """
#     Send an HTML email via SMTP.

#     Args:
#         to: Single address or list of addresses.
#         subject: Email subject line.
#         body_html: HTML body content.
#         body_text: Plain-text fallback (auto-stripped if not provided).
#         reply_to: Optional Reply-To header.

#     Returns:
#         Dict with status and message.
#     """
#     if isinstance(to, str):
#         to = [to]

#     msg = MIMEMultipart("alternative")
#     msg["Subject"] = subject
#     msg["From"] = f"{settings.smtp_from_name} <{settings.smtp_user}>"
#     msg["To"] = ", ".join(to)
#     if reply_to:
#         msg["Reply-To"] = reply_to

#     # Fallback plain text
#     plain = body_text or _html_to_plain(body_html)
#     msg.attach(MIMEText(plain, "plain"))
#     msg.attach(MIMEText(body_html, "html"))

#     try:
#         await aiosmtplib.send(
#             msg,
#             hostname=settings.smtp_host,
#             port=settings.smtp_port,
#             username=settings.smtp_user,
#             password=settings.smtp_pass,
#             start_tls=True,
#         )
#         logger.info("Email sent to %s: %s", to, subject)
#         return {"status": "sent", "recipients": to, "subject": subject}
#     except Exception as exc:
#         logger.error("Email send failed to %s: %s", to, exc)
#         return {"status": "failed", "error": str(exc), "recipients": to}


# async def send_support_reply(
#     customer_email: str,
#     customer_name: str,
#     reply_text: str,
#     original_subject: str = "",
# ) -> Dict:
#     """Send a support reply to a customer."""
#     subject = f"Re: {original_subject}" if original_subject else "Your PaddleAurum Support Request"
#     html = f"""
#     <html><body>
#     <p>Hi {customer_name or 'there'},</p>
#     <p>{reply_text.replace(chr(10), '<br>')}</p>
#     <br>
#     <p>🏓 Best regards,<br>
#     <strong>PaddleAurum Support Team</strong><br>
#     <a href="https://paddleaurum.com">paddleaurum.com</a></p>
#     </body></html>
#     """
#     return await send_email(customer_email, subject, html)


# async def send_promo_campaign(
#     recipients: List[str],
#     subject: str,
#     headline: str,
#     body: str,
#     cta_text: str,
#     cta_url: str,
#     discount_code: Optional[str] = None,
# ) -> Dict:
#     """Send a promotional email campaign to a list of recipients."""
#     discount_block = ""
#     if discount_code:
#         discount_block = f"""
#         <div style="background:#f0f9ff;border:2px dashed #0077cc;padding:16px;
#                     text-align:center;margin:24px 0;border-radius:8px;">
#           <strong>Use code: {discount_code}</strong> at checkout
#         </div>
#         """
#     html = f"""
#     <html><body style="font-family:Arial,sans-serif;max-width:600px;margin:auto;">
#     <div style="background:#1a1a2e;padding:20px;text-align:center;">
#       <h1 style="color:#ffd700;margin:0;">🏓 PaddleAurum</h1>
#     </div>
#     <div style="padding:24px;">
#       <h2 style="color:#1a1a2e;">{headline}</h2>
#       <p style="line-height:1.6;color:#333;">{body.replace(chr(10), '<br>')}</p>
#       {discount_block}
#       <div style="text-align:center;margin:32px 0;">
#         <a href="{cta_url}" style="background:#ffd700;color:#1a1a2e;padding:14px 32px;
#                                    text-decoration:none;border-radius:6px;
#                                    font-weight:bold;font-size:16px;">
#           {cta_text}
#         </a>
#       </div>
#     </div>
#     <div style="background:#f5f5f5;padding:12px;text-align:center;font-size:12px;color:#666;">
#       PaddleAurum | <a href="https://paddleaurum.com/unsubscribe">Unsubscribe</a>
#     </div>
#     </body></html>
#     """
#     results = []
#     # Send in batches of 50 to respect SMTP limits
#     for i in range(0, len(recipients), 50):
#         batch = recipients[i : i + 50]
#         result = await send_email(batch, subject, html)
#         results.append(result)

#     success_count = sum(1 for r in results if r["status"] == "sent")
#     return {
#         "status": "completed",
#         "batches_sent": len(results),
#         "success_batches": success_count,
#         "total_recipients": len(recipients),
#     }


# async def send_alert(subject: str, message: str) -> Dict:
#     """Send an operational alert to the admin email."""
#     html = f"""
#     <html><body>
#     <h2>⚠️ PaddleAurum Agent Alert</h2>
#     <p><strong>{subject}</strong></p>
#     <pre style="background:#f5f5f5;padding:12px;border-radius:4px;">{message}</pre>
#     </body></html>
#     """
#     return await send_email(settings.alert_email, f"[ALERT] {subject}", html)


# async def send_restock_alert(product_name: str, current_qty: int, reorder_qty: int, supplier_url: str) -> Dict:
#     """Notify admin of a restock requirement."""
#     subject = f"Restock Required: {product_name}"
#     message = (
#         f"Product: {product_name}\n"
#         f"Current quantity: {current_qty}\n"
#         f"Recommended reorder quantity: {reorder_qty}\n"
#         f"Supplier: {supplier_url}"
#     )
#     return await send_alert(subject, message)


# def _html_to_plain(html: str) -> str:
#     """Naive HTML → plain text stripping for fallback."""
#     import re
#     text = re.sub(r"<[^>]+>", " ", html)
#     text = re.sub(r"\s+", " ", text)
#     return text.strip()


# # ─────────────────────────────────────────────────────────────────────────────
# # LangChain BaseTool wrapper
# # ─────────────────────────────────────────────────────────────────────────────

# from typing import Dict   # noqa: E402 (already imported above but needed for annotation)


# class EmailTool(BaseTool):
#     name: str = "email_sender"
#     description: str = (
#         "Send emails for support replies, promotional campaigns, and operational alerts. "
#         "Actions: support_reply, promo_campaign, alert, restock_alert. "
#         "Input: JSON string with 'action' key and relevant parameters."
#     )

#     async def _arun(self, query: str) -> str:
#         try:
#             params = json.loads(query)
#             action = params.get("action")

#             if action == "support_reply":
#                 result = await send_support_reply(
#                     params["customer_email"],
#                     params.get("customer_name", ""),
#                     params["reply_text"],
#                     params.get("original_subject", ""),
#                 )
#             elif action == "promo_campaign":
#                 result = await send_promo_campaign(
#                     params["recipients"],
#                     params["subject"],
#                     params["headline"],
#                     params["body"],
#                     params["cta_text"],
#                     params["cta_url"],
#                     params.get("discount_code"),
#                 )
#             elif action == "alert":
#                 result = await send_alert(params["subject"], params["message"])
#             elif action == "restock_alert":
#                 result = await send_restock_alert(
#                     params["product_name"],
#                     params["current_qty"],
#                     params["reorder_qty"],
#                     params["supplier_url"],
#                 )
#             else:
#                 return json.dumps({"error": f"Unknown action: {action}"})

#             return json.dumps(result)
#         except Exception as exc:
#             logger.error("EmailTool error: %s", exc)
#             return json.dumps({"error": str(exc)})

#     def _run(self, query: str) -> str:
#         raise NotImplementedError("Use async _arun only")
































#@#@ @#@############################################################################################






















"""
tools/email_tool.py
───────────────────
Async email sending via aiosmtplib (SMTP) with connection pooling.
Used by Customer Captain (support replies), Promo General (campaigns),
and the error recovery node (critical alerts).
"""
from __future__ import annotations

import asyncio
import html
import json
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, List, Optional

import aiosmtplib
from langchain.tools import BaseTool

from config.settings import settings

logger = logging.getLogger(__name__)

# ── Connection pool ──────────────────────────────────────────────────────────

_smtp_connection: Optional[aiosmtplib.SMTP] = None
_smtp_lock = asyncio.Lock()


async def _get_smtp_connection() -> aiosmtplib.SMTP:
    """Return a persistent SMTP connection, reconnecting if necessary."""
    global _smtp_connection
    async with _smtp_lock:
        if _smtp_connection is None:
            client = aiosmtplib.SMTP(hostname=settings.smtp_host, port=settings.smtp_port)
            await client.connect()
            await client.starttls()
            await client.login(settings.smtp_user, settings.smtp_pass)
            _smtp_connection = client
            logger.debug("SMTP connection established")
        else:
            # Quick check: try NOOP to see if connection is alive
            try:
                await _smtp_connection.noop()
            except Exception:
                logger.warning("SMTP connection dead, reconnecting")
                await _smtp_connection.quit()
                client = aiosmtplib.SMTP(hostname=settings.smtp_host, port=settings.smtp_port)
                await client.connect()
                await client.starttls()
                await client.login(settings.smtp_user, settings.smtp_pass)
                _smtp_connection = client
        return _smtp_connection


async def _close_smtp_connection():
    """Close the persistent connection (used during shutdown)."""
    global _smtp_connection
    async with _smtp_lock:
        if _smtp_connection:
            await _smtp_connection.quit()
            _smtp_connection = None
            logger.debug("SMTP connection closed")


# ── HTML to plain text converter (robust) ───────────────────────────────────

class _HTMLToTextParser(html.parser.HTMLParser):
    """Simple HTML to plain text converter using built-in HTMLParser."""
    def __init__(self):
        super().__init__()
        self._text = []
        self._ignore = False

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'):
            self._ignore = True

    def handle_endtag(self, tag):
        if tag in ('script', 'style'):
            self._ignore = False
        elif tag in ('p', 'br', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'tr', 'li'):
            self._text.append('\n')

    def handle_data(self, data):
        if not self._ignore and data.strip():
            self._text.append(data.strip())

    def get_text(self) -> str:
        return ' '.join(self._text)


def _html_to_plain(html_content: str) -> str:
    """Convert HTML to plain text using a robust parser."""
    parser = _HTMLToTextParser()
    parser.feed(html_content)
    return parser.get_text()


# ── Core send function ──────────────────────────────────────────────────────

async def send_email(
    to: str | List[str],
    subject: str,
    body_html: str,
    body_text: Optional[str] = None,
    reply_to: Optional[str] = None,
) -> Dict:
    """
    Send an HTML email via SMTP using a persistent connection.

    Args:
        to: Single address or list of addresses.
        subject: Email subject line.
        body_html: HTML body content.
        body_text: Plain-text fallback (auto‑generated if not provided).
        reply_to: Optional Reply-To header.

    Returns:
        Dict with status and message.
    """
    if isinstance(to, str):
        to = [to]

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"{settings.smtp_from_name} <{settings.smtp_user}>"
    msg["To"] = ", ".join(to)
    if reply_to:
        msg["Reply-To"] = reply_to

    # Fallback plain text
    plain = body_text or _html_to_plain(body_html)
    msg.attach(MIMEText(plain, "plain"))
    msg.attach(MIMEText(body_html, "html"))

    try:
        conn = await _get_smtp_connection()
        await conn.send_message(msg)
        logger.info("Email sent to %s: %s", to, subject)
        return {"status": "sent", "recipients": to, "subject": subject}
    except Exception as exc:
        logger.error("Email send failed to %s: %s", to, exc)
        # Mark connection as dead so next call reconnects
        global _smtp_connection
        async with _smtp_lock:
            if _smtp_connection:
                await _smtp_connection.quit()
                _smtp_connection = None
        return {"status": "failed", "error": str(exc), "recipients": to}


async def send_support_reply(
    customer_email: str,
    customer_name: str,
    reply_text: str,
    original_subject: str = "",
) -> Dict:
    """Send a support reply to a customer."""
    subject = f"Re: {original_subject}" if original_subject else "Your PaddleAurum Support Request"
    html = f"""
    <html><body>
    <p>Hi {customer_name or 'there'},</p>
    <p>{reply_text.replace(chr(10), '<br>')}</p>
    <br>
    <p>🏓 Best regards,<br>
    <strong>PaddleAurum Support Team</strong><br>
    <a href="https://paddleaurum.com">paddleaurum.com</a></p>
    </body></html>
    """
    return await send_email(customer_email, subject, html)


async def send_promo_campaign(
    recipients: List[str],
    subject: str,
    headline: str,
    body: str,
    cta_text: str,
    cta_url: str,
    discount_code: Optional[str] = None,
) -> Dict:
    """Send a promotional email campaign to a list of recipients."""
    discount_block = ""
    if discount_code:
        discount_block = f"""
        <div style="background:#f0f9ff;border:2px dashed #0077cc;padding:16px;
                    text-align:center;margin:24px 0;border-radius:8px;">
          <strong>Use code: {discount_code}</strong> at checkout
        </div>
        """
    html = f"""
    <html><body style="font-family:Arial,sans-serif;max-width:600px;margin:auto;">
    <div style="background:#1a1a2e;padding:20px;text-align:center;">
      <h1 style="color:#ffd700;margin:0;">🏓 PaddleAurum</h1>
    </div>
    <div style="padding:24px;">
      <h2 style="color:#1a1a2e;">{headline}</h2>
      <p style="line-height:1.6;color:#333;">{body.replace(chr(10), '<br>')}</p>
      {discount_block}
      <div style="text-align:center;margin:32px 0;">
        <a href="{cta_url}" style="background:#ffd700;color:#1a1a2e;padding:14px 32px;
                                   text-decoration:none;border-radius:6px;
                                   font-weight:bold;font-size:16px;">
          {cta_text}
        </a>
      </div>
    </div>
    <div style="background:#f5f5f5;padding:12px;text-align:center;font-size:12px;color:#666;">
      PaddleAurum | <a href="https://paddleaurum.com/unsubscribe">Unsubscribe</a>
    </div>
    </body></html>
    """
    results = []
    # Send in batches of 50 to respect SMTP limits
    for i in range(0, len(recipients), 50):
        batch = recipients[i : i + 50]
        result = await send_email(batch, subject, html)
        results.append(result)

    success_count = sum(1 for r in results if r["status"] == "sent")
    return {
        "status": "completed",
        "batches_sent": len(results),
        "success_batches": success_count,
        "total_recipients": len(recipients),
    }


async def send_alert(subject: str, message: str) -> Dict:
    """Send an operational alert to the admin email."""
    html = f"""
    <html><body>
    <h2>⚠️ PaddleAurum Agent Alert</h2>
    <p><strong>{subject}</strong></p>
    <pre style="background:#f5f5f5;padding:12px;border-radius:4px;">{message}</pre>
    </body></html>
    """
    return await send_email(settings.alert_email, f"[ALERT] {subject}", html)


async def send_restock_alert(product_name: str, current_qty: int, reorder_qty: int, supplier_url: str) -> Dict:
    """Notify admin of a restock requirement."""
    subject = f"Restock Required: {product_name}"
    message = (
        f"Product: {product_name}\n"
        f"Current quantity: {current_qty}\n"
        f"Recommended reorder quantity: {reorder_qty}\n"
        f"Supplier: {supplier_url}"
    )
    return await send_alert(subject, message)


# ─────────────────────────────────────────────────────────────────────────────
# LangChain BaseTool wrapper
# ─────────────────────────────────────────────────────────────────────────────

class EmailTool(BaseTool):
    name: str = "email_sender"
    description: str = (
        "Send emails for support replies, promotional campaigns, and operational alerts. "
        "Actions: support_reply, promo_campaign, alert, restock_alert. "
        "Input: JSON string with 'action' key and relevant parameters."
    )

    async def _arun(self, query: str) -> str:
        try:
            params = json.loads(query)
            action = params.get("action")

            if action == "support_reply":
                result = await send_support_reply(
                    params["customer_email"],
                    params.get("customer_name", ""),
                    params["reply_text"],
                    params.get("original_subject", ""),
                )
            elif action == "promo_campaign":
                result = await send_promo_campaign(
                    params["recipients"],
                    params["subject"],
                    params["headline"],
                    params["body"],
                    params["cta_text"],
                    params["cta_url"],
                    params.get("discount_code"),
                )
            elif action == "alert":
                result = await send_alert(params["subject"], params["message"])
            elif action == "restock_alert":
                result = await send_restock_alert(
                    params["product_name"],
                    params["current_qty"],
                    params["reorder_qty"],
                    params["supplier_url"],
                )
            else:
                return json.dumps({"error": f"Unknown action: {action}"})

            return json.dumps(result)
        except Exception as exc:
            logger.error("EmailTool error: %s", exc)
            return json.dumps({"error": str(exc)})

    def _run(self, query: str) -> str:
        raise NotImplementedError("Use async _arun only")
        
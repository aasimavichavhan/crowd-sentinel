"""
Crowd Sentinel: Automated Email Alert Module
=============================================
Sends email alerts when crowd risk reaches Dense or Risky tiers.
Uses SMTP (Gmail App Password) when configured, otherwise logs to console.

Configuration via environment variables:
  SMTP_HOST       - SMTP server (default: smtp.gmail.com)
  SMTP_PORT       - SMTP port (default: 587)
  SMTP_USER       - Sender email address
  SMTP_PASSWORD   - App password (NOT your Google account password)
  ALERT_RECIPIENT - Recipient email (default: tejaskedarpawar@gmail.com)
"""

import os
import time
import smtplib
import threading
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, Any, Optional, List

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", 587))
SMTP_USER = os.environ.get("SMTP_USER", "")           # Set your Gmail address
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")    # Set your App Password
ALERT_RECIPIENT = os.environ.get("ALERT_RECIPIENT", "tejaskedarpawar@gmail.com")

# Minimum seconds between email dispatches (prevents flooding)
EMAIL_COOLDOWN_SECONDS = 60


class EmailAlerter:
    """
    Thread-safe automated email alerter.
    Sends HTML-formatted crowd risk alerts when overall tier is Dense or Risky.
    """

    def __init__(self, recipient: str = ALERT_RECIPIENT, cooldown: float = EMAIL_COOLDOWN_SECONDS):
        self.recipient = recipient
        self.cooldown = cooldown
        self._last_email_time: float = 0.0
        self._lock = threading.Lock()
        self._email_log: List[Dict[str, Any]] = []
        self._smtp_configured = bool(SMTP_USER and SMTP_PASSWORD)

        if self._smtp_configured:
            print(f"📧 Email Alerter: SMTP configured → {SMTP_USER} → {self.recipient}")
        else:
            print(f"📧 Email Alerter: SMTP NOT configured (console-log mode)")
            print(f"   Set SMTP_USER and SMTP_PASSWORD env vars for live email delivery.")

    @property
    def is_smtp_ready(self) -> bool:
        return self._smtp_configured

    @property
    def email_log(self) -> List[Dict[str, Any]]:
        """Returns the most recent email dispatch log entries."""
        return self._email_log[-20:]

    def should_send(self) -> bool:
        """Check if enough time has passed since the last email."""
        return (time.time() - self._last_email_time) >= self.cooldown

    def try_send_alert(
        self,
        overall_tier: str,
        average_risk: float,
        zones: List[Dict[str, Any]],
        new_alerts: List[Dict[str, Any]],
        total_people: int
    ) -> Optional[Dict[str, Any]]:
        """
        Checks if the tier is Dense or Risky and sends an email alert.
        Returns a dispatch log entry if sent, None otherwise.
        """
        # Only trigger on Dense or Risky
        if overall_tier not in ("Dense", "Risky"):
            return None

        with self._lock:
            if not self.should_send():
                return None
            self._last_email_time = time.time()

        # Build the email
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        subject = f"🚨 CROWD SENTINEL ALERT: {overall_tier.upper()} Risk Detected!"

        # Gather critical zone details
        critical_zones = [z for z in zones if z.get("risk_level") in ("Dense", "Risky")]
        zone_rows_html = ""
        for z in critical_zones:
            tier_color = "#ef4444" if z["risk_level"] == "Risky" else "#f97316"
            zone_rows_html += f"""
            <tr>
                <td style="padding:8px 12px;border-bottom:1px solid #2d3748;">{z['zone_id']}</td>
                <td style="padding:8px 12px;border-bottom:1px solid #2d3748;">{z.get('name', z['zone_id'])}</td>
                <td style="padding:8px 12px;border-bottom:1px solid #2d3748;color:{tier_color};font-weight:700;">{z['risk_level']}</td>
                <td style="padding:8px 12px;border-bottom:1px solid #2d3748;">{z['count']} pax</td>
                <td style="padding:8px 12px;border-bottom:1px solid #2d3748;">{z['turbulence_score']}%</td>
                <td style="padding:8px 12px;border-bottom:1px solid #2d3748;">{z['risk_score']}/100</td>
            </tr>"""

        alert_messages_html = ""
        for a in new_alerts:
            alert_messages_html += f"""
            <div style="background:#2d1b1b;border-left:4px solid #ef4444;padding:10px 14px;margin-bottom:8px;border-radius:4px;">
                <strong style="color:#fca5a5;">[{a.get('timestamp', timestamp)}]</strong>
                <span style="color:#fde8e8;">{a.get('message', 'Risk alert triggered')}</span>
            </div>"""

        if not alert_messages_html:
            alert_messages_html = f"""
            <div style="background:#2d2b1b;border-left:4px solid #f97316;padding:10px 14px;border-radius:4px;">
                <strong style="color:#fcd34d;">Automated Detection</strong>
                <span style="color:#fef3c7;">Overall crowd risk has escalated to <strong>{overall_tier}</strong>.</span>
            </div>"""

        tier_bg = "#dc2626" if overall_tier == "Risky" else "#ea580c"

        html_body = f"""
        <div style="font-family:'Segoe UI',Arial,sans-serif;max-width:640px;margin:0 auto;background:#0f172a;color:#e2e8f0;border-radius:12px;overflow:hidden;">
            <!-- Header Banner -->
            <div style="background:{tier_bg};padding:20px 24px;text-align:center;">
                <h1 style="margin:0;font-size:22px;color:white;letter-spacing:1px;">🛡️ CROWD SENTINEL — {overall_tier.upper()} ALERT</h1>
                <p style="margin:6px 0 0;color:rgba(255,255,255,0.85);font-size:13px;">Automated Stampede Early-Warning Notification</p>
            </div>

            <!-- Summary -->
            <div style="padding:20px 24px;">
                <table style="width:100%;border-collapse:collapse;margin-bottom:16px;">
                    <tr>
                        <td style="padding:6px 0;color:#94a3b8;font-size:13px;">Timestamp</td>
                        <td style="padding:6px 0;font-weight:600;">{timestamp}</td>
                    </tr>
                    <tr>
                        <td style="padding:6px 0;color:#94a3b8;font-size:13px;">Overall Risk Tier</td>
                        <td style="padding:6px 0;font-weight:700;color:{tier_bg};">{overall_tier}</td>
                    </tr>
                    <tr>
                        <td style="padding:6px 0;color:#94a3b8;font-size:13px;">Average Risk Score</td>
                        <td style="padding:6px 0;font-weight:600;">{average_risk}/100</td>
                    </tr>
                    <tr>
                        <td style="padding:6px 0;color:#94a3b8;font-size:13px;">Total People Detected</td>
                        <td style="padding:6px 0;font-weight:600;">{total_people}</td>
                    </tr>
                </table>

                <!-- Alert Messages -->
                <h3 style="color:#f8fafc;font-size:14px;margin:16px 0 8px;text-transform:uppercase;letter-spacing:0.5px;">⚠️ Active Alerts</h3>
                {alert_messages_html}

                <!-- Critical Zones Table -->
                <h3 style="color:#f8fafc;font-size:14px;margin:20px 0 8px;text-transform:uppercase;letter-spacing:0.5px;">📊 Critical Zone Breakdown</h3>
                <table style="width:100%;border-collapse:collapse;font-size:13px;background:#1e293b;border-radius:6px;overflow:hidden;">
                    <thead>
                        <tr style="background:#334155;">
                            <th style="padding:8px 12px;text-align:left;color:#94a3b8;">Zone</th>
                            <th style="padding:8px 12px;text-align:left;color:#94a3b8;">Name</th>
                            <th style="padding:8px 12px;text-align:left;color:#94a3b8;">Risk</th>
                            <th style="padding:8px 12px;text-align:left;color:#94a3b8;">Count</th>
                            <th style="padding:8px 12px;text-align:left;color:#94a3b8;">Turbulence</th>
                            <th style="padding:8px 12px;text-align:left;color:#94a3b8;">Score</th>
                        </tr>
                    </thead>
                    <tbody>
                        {zone_rows_html if zone_rows_html else '<tr><td colspan="6" style="padding:12px;text-align:center;color:#64748b;">No individual zone at critical level</td></tr>'}
                    </tbody>
                </table>

                <!-- Footer -->
                <div style="margin-top:24px;padding-top:16px;border-top:1px solid #1e293b;text-align:center;">
                    <p style="font-size:11px;color:#64748b;margin:0;">
                        This is an automated alert from Crowd Sentinel.<br>
                        Privacy: Zero biometrics — spatial density &amp; optical flow vectors only.
                    </p>
                </div>
            </div>
        </div>
        """

        dispatch_entry = {
            "timestamp": timestamp,
            "tier": overall_tier,
            "average_risk": average_risk,
            "total_people": total_people,
            "critical_zones": len(critical_zones),
            "recipient": self.recipient,
            "method": "SMTP" if self._smtp_configured else "CONSOLE_LOG"
        }

        # Send in a background thread so it doesn't block the stream loop
        thread = threading.Thread(
            target=self._dispatch_email,
            args=(subject, html_body, dispatch_entry),
            daemon=True
        )
        thread.start()

        return dispatch_entry

    def _dispatch_email(self, subject: str, html_body: str, log_entry: Dict[str, Any]):
        """Actually send the email (or log to console)."""
        try:
            if self._smtp_configured:
                msg = MIMEMultipart("alternative")
                msg["Subject"] = subject
                msg["From"] = SMTP_USER
                msg["To"] = self.recipient
                msg.attach(MIMEText(html_body, "html"))

                with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
                    server.ehlo()
                    server.starttls()
                    server.ehlo()
                    server.login(SMTP_USER, SMTP_PASSWORD)
                    server.sendmail(SMTP_USER, [self.recipient], msg.as_string())

                log_entry["status"] = "SENT"
                print(f"📧 Email SENT to {self.recipient}: {subject}")
            else:
                # Console-log mode (no SMTP credentials)
                log_entry["status"] = "LOGGED"
                print(f"\n{'='*70}")
                print(f"📧 EMAIL ALERT (console mode — SMTP not configured)")
                print(f"   To:      {self.recipient}")
                print(f"   Subject: {subject}")
                print(f"   Tier:    {log_entry['tier']} | Risk: {log_entry['average_risk']}/100")
                print(f"   People:  {log_entry['total_people']} | Critical Zones: {log_entry['critical_zones']}")
                print(f"   Time:    {log_entry['timestamp']}")
                print(f"{'='*70}\n")

        except Exception as e:
            log_entry["status"] = "FAILED"
            log_entry["error"] = str(e)
            print(f"❌ Email dispatch FAILED: {e}")

        self._email_log.append(log_entry)

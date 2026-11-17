"""Read-only bosdyn session (no lease is taken, so the tablet can keep driving)."""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)


def connect(ip: str | None = None, username: str | None = None, password: str | None = None,
            client_name: str = "spot_calib"):
    import bosdyn.client

    ip = ip or os.environ.get("ADT4_BOSDYN_IP")
    username = username or os.environ.get("ADT4_BOSDYN_USERNAME")
    password = password or os.environ.get("ADT4_BOSDYN_PASSWORD")
    if not (ip and username and password):
        raise RuntimeError("Spot credentials missing: set ADT4_BOSDYN_IP / ADT4_BOSDYN_USERNAME / "
                           "ADT4_BOSDYN_PASSWORD or pass --spot-ip/--spot-user/--spot-password")
    sdk = bosdyn.client.create_standard_sdk(client_name)
    robot = sdk.create_robot(ip)
    robot.authenticate(username, password)
    try:
        robot.time_sync.wait_for_sync(10)
    except Exception as e:  # time sync is only nice-to-have here
        log.warning("Spot time sync failed: %s", e)
    log.info("Connected to Spot at %s", ip)
    return robot

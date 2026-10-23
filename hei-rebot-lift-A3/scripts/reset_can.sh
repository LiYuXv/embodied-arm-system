#!/bin/bash
# Reset gs_usb (candlelight) CAN adapters and bring up CAN interfaces.
# Needed after an unexpected power loss, where the adapter firmware hangs
# until the USB device is re-enumerated (simulates unplug/replug).
#
# Self-healing: if bringing an interface up fails (e.g. "Device or resource
# busy"), the interface is released (set down) and retried; after repeated
# failure the adapter's USB device is re-authorized (virtual replug).
#
# Must be run as root.

BITRATE=${BITRATE:-1000000}
IFACES=${IFACES:-"can0 can1"}

modprobe can
modprobe can_raw
modprobe gs_usb

# Deauthorize/reauthorize all gs_usb adapters (1d50:606f) = virtual replug
reset_adapters() {
    for dev in /sys/bus/usb/devices/*; do
        [ -f "$dev/idVendor" ] || continue
        if [ "$(cat "$dev/idVendor")" = "1d50" ] && [ "$(cat "$dev/idProduct")" = "606f" ]; then
            echo "Resetting USB device: $dev"
            echo 0 > "$dev/authorized"
            sleep 0.5
            echo 1 > "$dev/authorized"
        fi
    done
    sleep 2
}

# Wait up to ~30s for an interface to appear
wait_iface() {
    local iface="$1"
    for attempt in $(seq 1 15); do
        if ip link show "$iface" &>/dev/null; then
            return 0
        fi
        sleep 2
    done
    echo "Error: $iface did not appear"
    return 1
}

# Bring up an interface; on failure release (down) and retry
bring_up() {
    local iface="$1"
    for attempt in $(seq 1 5); do
        ip link set "$iface" down 2>/dev/null
        sleep 0.2
        if ip link set "$iface" up type can bitrate "$BITRATE" 2>/dev/null; then
            echo "$iface up @ $BITRATE bps"
            return 0
        fi
        echo "$iface busy/failed, releasing and retrying ($attempt/5)"
        sleep 1
    done
    return 1
}

reset_adapters

failed=0
for iface in $IFACES; do
    if wait_iface "$iface" && bring_up "$iface"; then
        continue
    fi
    failed=1
done

# Last resort: full virtual replug, then try whatever is still down
if [ "$failed" -eq 1 ]; then
    echo "Retrying with full adapter reset..."
    reset_adapters
    for iface in $IFACES; do
        if wait_iface "$iface" && bring_up "$iface"; then
            continue
        fi
        echo "Error: failed to bring up $iface"
    done
fi

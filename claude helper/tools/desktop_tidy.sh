#!/usr/bin/env bash
# Undo the temporary remote-serving setup on the training desktop once the scenes have been copied to the
# viewing machine (scripts/serve.py /bundle). Run inside WSL on the desktop:
#
#   bash ~/pinn/"claude helper/tools/desktop_tidy.sh"            # stop the tunnel and the server, remove temporary files
#   bash ~/pinn/"claude helper/tools/desktop_tidy.sh" --keep-server
#
# Then, in an *Administrator* PowerShell on the desktop, close the network openings made for the server:
#
#   Set-NetFirewallHyperVVMSetting -Name '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' -DefaultInboundAction Block
#   Remove-NetFirewallRule -DisplayName pinn-serve
#
# Kept on purpose: the DNS fix (/etc/resolv.conf with generateResolvConf = false in /etc/wsl.conf) and
# networkingMode=mirrored in %USERPROFILE%\.wslconfig. Mirrored networking needs the DNS fix; to undo both,
# remove the mirrored line, the [network] block in /etc/wsl.conf, run `wsl --shutdown`, and WSL regenerates
# resolv.conf on the next start.
set -u
cd "$(dirname "$0")/../.."
keep_server=0
[ "${1:-}" = "--keep-server" ] && keep_server=1

if pkill -f "cloudflared tunnel" 2>/dev/null; then echo "stopped the Cloudflare tunnel"; else echo "no tunnel running"; fi
if [ "$keep_server" -eq 0 ]; then
  if pkill -f scripts/serve.py 2>/dev/null; then echo "stopped scripts/serve.py"; else echo "no server running"; fi
fi
rm -f cloudflared runs/tunnel.log && echo "removed the cloudflared binary and its log"
rm -rf runs/_bundle && echo "removed runs/_bundle (the copied bundle; rebuild any time via /bundle)"
git status --short
echo "now run the two Administrator PowerShell lines printed at the top of this script"

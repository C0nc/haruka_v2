#!/usr/bin/env bash
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
manifest="$here/download_manifest.tsv"
raw_root="$here/data/raw"
log="$raw_root/download.log"
mkdir -p "$raw_root/images" "$raw_root/plaque_masks"

tail -n +2 "$manifest" | while IFS=$'\t' read -r kind record filename expected_md5 url; do
    if [[ "$kind" == image ]]; then
        destination="$raw_root/images/$filename"
    else
        destination="$raw_root/plaque_masks/$filename"
    fi
    if [[ -f "$destination" ]] && [[ "$(md5sum "$destination" | cut -d' ' -f1)" == "$expected_md5" ]]; then
        printf '%s SKIP verified %s\n' "$(date -Is)" "$destination" >> "$log"
        continue
    fi
    printf '%s START %s record=%s\n' "$(date -Is)" "$destination" "$record" >> "$log"
    if curl -L --fail --retry 30 --retry-all-errors --retry-delay 10 \
        --connect-timeout 30 -C - --output "$destination" "$url" >> "$log" 2>&1; then
        actual_md5="$(md5sum "$destination" | cut -d' ' -f1)"
        if [[ "$actual_md5" == "$expected_md5" ]]; then
            printf '%s DONE verified %s\n' "$(date -Is)" "$destination" >> "$log"
        else
            printf '%s ERROR md5 expected=%s actual=%s %s\n' \
                "$(date -Is)" "$expected_md5" "$actual_md5" "$destination" >> "$log"
            exit 2
        fi
    else
        printf '%s ERROR download %s\n' "$(date -Is)" "$destination" >> "$log"
        exit 1
    fi
done

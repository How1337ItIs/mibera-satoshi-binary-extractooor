# CURSOR AGENT: BINARY DECODING & PATTERN SEARCH – FULL DOCUMENTATION
**Date:** July 16, 2025  
**Agent:** Cursor Agent  
**Phase:** Comprehensive Binary Decoding & Pattern Search

---

## 📋 **PROJECT CONTEXT & RATIONALE**

- **Goal:** Extract and analyze the 1,440-bit stream from the Satoshi Nakamoto poster for any hidden information, cryptographic keys, or meaningful patterns.
- **Why this phase?**
  - The bitstream is 100% coverage, validated, and has a highly structured ratio (75.6% zeros, 24.4% ones), suggesting possible padding or encoding rather than pure randomness.
  - Quick, methodical decoding passes are computationally cheap and can immediately reveal hidden text, keys, or encodings.
  - If nothing is found, the negative result is itself valuable, guiding the next phase (crypto-statistical analysis).

---

## 🧭 **STRATEGY & PLAN**

### **Why start with binary decoding?**
- The bitstream is validated and complete.
- The structure suggests possible encoding, not pure entropy.
- Fast, systematic decoding attempts can catch low-hanging fruit (text, keys, QR, etc.) before deeper statistical or cryptanalytic work.

### **Stepwise Plan (as discussed):**
| Step | What we do | Outcome / Signal |
|------|------------|------------------|
| 1 | Canonical byte split (row-major, 8-bit, no parity) | Fast sanity check for obvious text (e.g., “The Times …”) |
| 2 | All eight bit-shifts (0-7) of the entire stream | Catches misalignment; log printable byte ratios & hex dumps |
| 3 | Reverse-bit-order per byte & repeat shifts | Some stego hides bits LSB-first |
| 4 | Endianness swaps – little-/big-endian 16-bit chunks | Quick win if author packed words |
| 5 | Base64 / Base58 window scan – slide 6-bit & 5-bit windows, test decode | Flag windows that produce valid printable strings or version bytes |
| 6 | WIF / key prefix brute – scan 37- to 52-char Base58 windows for valid checksum | Immediate alert if we see version-prefix 0x80 / 0x00 keys |
| 7 | QR-matrix guess – reshape 37×37, 29×29, etc., feed to pyzbar | Cheap; fails fast if density wrong |
| 8 | Entropy drop monitor – every decode attempt logs Shannon entropy; sudden dip = structured payload |

- **Success criteria:**
  1. Any decoded string ≥ 20 printable chars
  2. Any valid Base58Check payload (checksum passes)
  3. Entropy dip ≥ 0.2 bits between raw bytes and decoded/pruned bytes

---

## 🛠️ **TOOLING & IMPLEMENTATION**

- **Script:** `scripts/decode_hunt.py` implements all steps above, with:
  - Logging to `decode_attempts.log`
  - Results to `candidate_hits.csv`, `decode_attempts.json`, and `decode_heatmap.png`
  - QR matrix attempts for plausible sizes
  - Entropy monitoring for all decode attempts
- **Dependencies:** All required packages listed in `requirements.txt` (numpy, pandas, matplotlib, pycryptodome, pyzbar, etc.)
- **Artefacts:**
  - `decode_attempts.log` – Full log of all decode attempts, offsets, printable ratios, entropy, and hits
  - `candidate_hits.csv` – Candidate hits (pending serialization fix)
  - `decode_attempts.json` – All decode attempts (JSON, for reproducibility)
  - `decode_heatmap.png` – Visual heatmap of printable ratio and entropy by offset

---

## 📝 **DISCUSSION & DECISION LOG**

- **Why not start with crypto-statistics?**
  - The bitstream’s structure and coverage make it ideal for quick decoding attempts first.
  - If nothing is found, we’ll have ruled out a large class of possible encodings and can focus on deeper analysis.
- **Why so many decoding variants?**
  - Many stego/crypto schemes use bit shifts, reversed bits, or non-canonical byte orderings.
  - Sliding window and endianness swaps catch misaligned or packed data.
- **Why QR matrix?**
  - Some puzzles encode data visually; a QR code is a cheap, fast check.
- **Why entropy monitoring?**
  - A sudden drop in entropy can signal a structured payload (e.g., compressed, encoded, or encrypted data block).

---

## 🕐 **EXECUTION TIMELINE**

### **[17:00] - Plan & Tooling**
- User provided a detailed, stepwise plan for binary decoding and pattern search.
- Steps include: byte splits, bit shifts, reverse bit order, endianness swaps, base64/base58 scans, WIF brute, QR matrix, entropy monitoring.
- Stub code and logging/artefact plan provided.

### **[17:05] - Tool Implementation**
- Created `scripts/decode_hunt.py` implementing all steps.
- Logging to `decode_attempts.log`, results to `candidate_hits.csv`, `decode_attempts.json`, and `decode_heatmap.png`.
- Updated `requirements.txt` for all dependencies.

### **[17:10] - First Run**
- Ran `python scripts/decode_hunt.py`.
- All steps executed; logging and heatmap generated.
- **WIF CANDIDATES** detected at several offsets (e.g., 312, 488, 536, 808, 848, 1008) with hex data starting with `0x80...`.
- QR matrix attempts for 37x37, 29x29, 25x25, 21x21 (black ratios: 0.247–0.463).
- No clear ASCII or base64/base58 hits.
- **Error:** CSV serialization failed due to non-serializable fields (`matrix`, `matrix_size`, `black_ratio`).

### **[17:15] - Bugfix Attempt**
- Patched `save_results()` to filter out non-serializable fields for CSV.
- Re-ran script; still failed due to additional fields (`black_ratio`, `matrix_size`).
- **Root Cause:** QR matrix results include fields not present in other candidate dicts, causing CSV field mismatch.

---

## 📊 **RESULTS & FINDINGS**

- **WIF Candidates:** Multiple offsets flagged as possible WIF (Wallet Import Format) private key candidates, all with version byte 0x80. No valid base58check or printable key detected yet.
- **QR Matrix:** No QR code detected, but black/white ratios logged for several plausible sizes.
- **ASCII/Base64/Base58:** No obvious hits; no long printable strings or known Bitcoin phrases found.
- **Entropy Monitoring:** Heatmap generated (`decode_heatmap.png`), no dramatic entropy dips observed.
- **Artefacts Generated:**
  - `decode_attempts.log` (full log of all decode attempts)
  - `decode_heatmap.png` (offset vs printable-ratio/entropy)
  - `candidate_hits.csv` (failed to write due to serialization bug)
  - `decode_attempts.json` (all decode attempts, not yet verified)

---

## 🛠️ **ISSUES & LESSONS LEARNED**

- **CSV Serialization Error:**
  - Non-uniform candidate dicts (QR matrix results have extra fields).
  - Solution: Filter all candidate dicts to a common set of fields before writing CSV.
- **No Decoding Hits Yet:**
  - No ASCII, base64, or base58 decodes yielded long printable strings or known phrases.
  - WIF candidates detected, but not validated as real keys.

---

## 📈 **NEXT STEPS**

1. **Fix CSV Serialization:**
   - Ensure all candidate dicts have a uniform set of fields for CSV output.
   - Optionally, split QR matrix results into a separate file.
2. **Review WIF Candidates:**
   - Check if any flagged WIF candidates are valid base58check keys.
   - Attempt to decode/validate as Bitcoin private keys.
3. **Review Artefacts:**
   - Examine `decode_heatmap.png` for any entropy dips or printable ratio spikes.
   - Inspect `decode_attempts.log` for any overlooked hits.
4. **If No Hits:**
   - Move to Direction #2: deeper crypto-statistical/randomness analysis.

---

## 🚦 **SUCCESS CRITERIA (RECAP)**
- Any decoded string ≥ 20 printable chars
- Any valid Base58Check payload (checksum passes)
- Entropy dip ≥ 0.2 bits between raw bytes and decoded/pruned bytes

---

## 🧑‍💻 **CODE SNIPPETS & EXAMPLES**

### **Stub for Canonical Byte Split & ASCII Scan**
```python
BITS = np.loadtxt("cursor_optimized_extraction.csv", delimiter=",", dtype=int)
bits = "".join(map(str, BITS))

def bits_to_bytes(bitstr, rev=False):
    if rev:
        bitstr = "".join(bs[::-1] for bs in textwrap.wrap(bitstr, 8))
    return bytes(int(bitstr[i:i+8], 2) for i in range(0, len(bitstr), 8))

def try_ascii(b):
    txt = "".join(chr(x) if 32 <= x <= 126 else "." for x in b)
    if any(w in txt for w in ("The", "Bitcoin", "Satoshi")):
        print("[ASCII HIT]", txt[:120])

for shift in range(8):
    s = bits[shift:]
    b = bits_to_bytes(s)
    try_ascii(b)
    b_rev = bits_to_bytes(s, rev=True)
    try_ascii(b_rev)
```

---

## 📁 **ARTEFACTS & OUTPUTS**
- `decode_attempts.log` – Full log of all decode attempts, offsets, printable ratios, entropy, and hits
- `candidate_hits.csv` – Candidate hits (pending serialization fix)
- `decode_attempts.json` – All decode attempts (JSON, for reproducibility)
- `decode_heatmap.png` – Visual heatmap of printable ratio and entropy by offset

---

## 🧭 **DECISION TREE & PIVOT LOGIC**
- If any step yields a hit (long printable string, valid key, entropy dip), stop and analyze that result in depth.
- If all steps fail, pivot to Direction #2: advanced crypto-statistical/randomness analysis.

---

## 🏁 **STATUS & NEXT ACTIONS**
- **Current status:** In progress, pending CSV serialization fix and review of WIF/QR/entropy artefacts.
- **Next actions:**
  1. Fix candidate CSV serialization
  2. Review WIF candidates for validity
  3. Review artefacts for missed hits
  4. If no hits, move to crypto-statistics phase

---

**This document is a living record of all context, rationale, technical steps, code, results, issues, and next actions for the binary decoding & pattern search phase.** 
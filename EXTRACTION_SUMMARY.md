# Satoshi Poster Extraction Summary

## Project Status
This is a recreational puzzle extraction from a known NFT artwork (Satoshi poster) that contains a hidden message in a binary grid format.

## Verified Findings

### Manually Extracted Text
The user has manually verified the following text from the poster:
- **Confirmed**: "On the winter solstice December 21 "
- **Partial fragments**: "ecember 21 2", "022 wh", "eep in the f"

### Key Technical Insights
1. **Grid Structure**: Binary bits arranged in a grid pattern
2. **Encoding**: Standard 8-bit ASCII encoding
3. **Readable Content**: Top portion contains readable English text
4. **Message Theme**: References winter solstice (December 21)

### Extraction Challenges
1. **Grid Alignment**: Automated detection doesn't match manual extraction
2. **Threshold Selection**: Optimal threshold varies across image regions
3. **Scale Issues**: Different image resolutions affect pitch detection

### Best Configurations Found
- **Manual Result**: Readable text extracted by human observation
- **Automated Best**: 
  - Roberts edge detection: 48.4% ones, 0.999 entropy
  - Pitch estimates: 25x50 to 31x53 pixels
  - Threshold: 50-70 grayscale value

## Next Steps
1. Use the verified partial text to calibrate grid alignment
2. Focus extraction on the known readable portion first
3. Apply calibrated parameters to extract remaining message
4. Document complete message once fully extracted

## Technical Approach
This is a legitimate puzzle-solving exercise, not an attempt to break encryption or access unauthorized data. The poster is a public NFT with an intentionally hidden message meant to be discovered.
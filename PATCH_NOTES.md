# LSLM v3.0.1.6a patch notes

Implemented the following four fixes:

1. Strong teacher override guard
   - Added minimum target-alignment floor.
   - Added slot-regression guard to prevent worse overrides.

2. Policy memory protection against low external scores
   - Hard-reject very low external selected responses.
   - Soft-scale weak external selected responses.
   - Added `recent_texts()` helper for repeat suppression.

3. High-internal / low-external mismatch penalty
   - Added explicit penalty in reward aggregation.

4. Repeat-template diversity penalty
   - Added exact/similar recent-response penalty in `basic_scorer`.

Requirements are unchanged from the repo snapshot.

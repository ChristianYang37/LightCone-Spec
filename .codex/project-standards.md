# Project standards

1. Every code change follows Occam's razor: choose the smallest clear solution; add no
   abstraction, configuration, check, dependency, or test without a concrete need.
2. Keep Target-only, Static, TTS, L0-naive, LightCone, and OnlineSPEC distinct.
3. Preserve exact sampling, proposal-version safety, finite updates, complete raw metrics,
   paired blocks, and pilot/final separation.
4. A failed or unsupported cell stays visible as failed or skipped.
5. Test only behavior and failure modes that matter; GPU smoke requires an authorized host.
6. Never gate research on cryptographic content identity or a clean Git tree.

# Methodology and defensible submission claims

The supplied PDF is a project-planning conversation. It describes a software-only
proposal combining YOLOv8, MobileNetV2 with dilated convolutions, pose/kinematics,
LSTM, weighted risk logic and an operator dashboard. It is not itself the source
code or independently verified experimental record of the original papers.

## What can be demonstrated now

- Real recorded video runs through person detection and temporary track IDs.
- A custom density head trained on real point annotations returns a count and map.
- A trained LSTM forecasts count pseudo-labels, with explicit baseline alternatives.
- The API worker publishes real results to a dashboard, with persisted alerts and CSV.
- Missing optional modules report their state; no random predictions fill the UI.

The current contribution is a reproducible **integrated software prototype**.
It is not a novel state-of-the-art detector, a finished behavioral classifier,
or demonstrated stampede prevention. Density estimation and person detection are
shown independently; no calibrated count-fusion estimator has been fitted.

## Experimental score, not a probability

Let occupancy be the detected count divided by a user-supplied reference capacity.
The implemented score is:

`70 * min(occupancy, 1) + 20 * motion + 10 * min(positive_forecast_growth / capacity, 1)`

`motion = min(1, 0.6 * reversal_fraction + 0.3 * sudden_motion_fraction + 0.1 * stalled_fraction * min(occupancy, 1))`.

All weights and image-space thresholds are exploratory, not learned incident
probabilities. Reversal means successive trajectory velocities form a sufficiently
opposite angle. Slow/fast thresholds use normalized image coordinates per second,
not metres/second. Camera movement, perspective, ID switches and occlusion can
invalidate these signals. Pose landmarks are optional visual output and do not
enter this version's risk score.

Rapid-dispersal evidence retains both a group mean-speed peak and an any-track
peak for five seconds. The sparse-track path requires a recent count of at least
five people, a drop of at least 50%, and peak normalized speed of at least 0.06;
it is intended for clips where tracking loses most IDs as people leave quickly.

Tier boundaries are 35/60/80 for Moderate/High/Critical. General motion evidence
must persist for 1.5 seconds of **video time** before alerting; rapid dispersal
uses a one-second rule. One brief tracking dropout does not reset persistence, and
two seconds of normal observations re-arms the zone for a new incident. A
30-second cooldown suppresses repeats during one continuing incident, while tier
escalation can bypass that cooldown. A scene-cut heuristic resets tracking,
forecast history and alert persistence. Cuts are not perfectly detected.
The policy is not calibrated for any venue. Without capacity, occupancy remains
uncalibrated, but a Moderate motion warning can be assigned when at least three
people are tracked and their mean normalized image-space speed is at least 0.04,
or related sudden/reversal thresholds are crossed. With capacity configured, the
same motion condition sets a Moderate score floor. These thresholds are exploratory,
not a trained action-recognition or incident classifier, and still require the
applicable temporal persistence rule before an alert is stored.

## Counting protocol

ShanghaiTech Part B official training images are split 80/20 with RNG seed 42.
The official test set is held out. Inputs are resized to 256x256 and normalized
with ImageNet channel statistics. A truncated pretrained MobileNetV2 provides
96-channel 16x16 features; its weights remain frozen. A 64/32/1-channel dilated
head with nonnegative output is trained against count-conserving Gaussian maps.
The objective combines scaled count MSE and spatial-map MSE. Validation MAE
selects the checkpoint. Test MAE and RMSE are computed afterward.

The constant training-mean comparison is a sanity baseline only. It cannot
establish superiority over SmartCrowd, the ESWA model described in the PDF,
CSRNet, or other published models with different protocols.

## Forecast protocol

YOLOv8n generates one count per second from the UMN university demonstration.
Frames are partitioned chronologically before creating eight-count input windows
and three-second targets. Training-only statistics normalize features; no window
crosses a partition or detected scene boundary. Validation MAE selects the model.
Report LSTM and last-observation persistence on the same test targets.

This split avoids overlapping cross-partition windows, but all partitions still
come from a single edited demonstration. It is not independent event-level
validation. YOLO errors enter the target labels. Counts derived by prediction are
not ground truth, and the LSTM's higher MAE must not be hidden.

## Work needed for the original accuracy goal

1. Obtain original base papers and their reproducible code/protocols; compare on
   identical data, units, splits, preprocessing and hardware.
2. Improve density estimation with training-only augmentation and validated
   scale/perspective handling. Reserve a fresh holdout after model selection.
3. Obtain multiple independently labelled, licensed videos, including the intended
   Indian-context conditions; split by recording/event, not shuffled frames.
4. Label incidents and normal periods under a written protocol. Do not infer
   pushing, panic, intent or danger from poses alone.
5. Measure precision/recall, PR-AUC, false alerts per hour, missed incidents and
   warning lead time. Counting MAE cannot substitute for these measures.
6. Compare occupancy-only, occupancy+kinematics, and forecasting ablations.
   Include uncertainty intervals, occlusion levels and low-light/domain-shift tests.
7. Report median/p95 end-to-end latency and throughput on the actual target laptop.
   A sampled-frame timing from this development runtime is not a laptop guarantee.

## Known implementation limits

One CPU video worker at a time; uploaded recordings up to 250 MB, 60 minutes and
4K; dashboard samples at two frames per second. Processing speed depends on
hardware and enabled models. The UI has full/split-frame zone presets; the API
accepts normalized polygons. Split presets share their exact boundary, so a
bottom-center point on that line can belong to both zones; do not sum zone counts
as a unique whole-scene total. Arbitrary overlapping API zones have the same rule.
Jobs interrupted by restart remain marked interrupted and need a fresh upload.
Only a bounded recent history is served by the API; see the storage implementation.
No trained action recognition, camera calibration, webcam/RTSP, audio, device
tracking, authentication, multitenancy, automatic retention, or public deployment.

Overview

A single knee scan can reveal a dozen different problems. In this competition, you are tasked to build machine learning models that detect a defined set of clinically important abnormalities on knee MRI examinations.

Start
2 days ago

Close
3 months to go
Merger & Entry
Description

The knee is the most commonly injured and imaged joint in the body. Osteoarthritis alone affects an estimated 654 million people worldwide, while acute knee injuries account for 15 to 40 percent of all sports-related trauma. MRIs show clinicians ligaments, cartilage, menisci, and bone in detail, without exposing patients to radiation.

Reading those scans isn’t always straightforward. ACL and MCL tears, meniscal damage, cartilage loss, fractures, and other abnormalities can be subtle, and radiologists don’t always interpret them the same way. Access to musculoskeletal radiologists is also limited, especially outside major medical centers, leading to delays and inconsistent diagnoses.

In this competition, you will develop multimodal machine learning models to detect twelve clinically important knee abnormalities. You'll work with the first RSNA AI Challenge dataset that pairs every imaging study with its original radiology report, enabling your models to learn from both visual scans and written diagnostic text.

High-performing models can act as robust decision support tools, delivering the accuracy, consistency, and speed needed to elevate expert-level knee MRI interpretation and improve care across disparate clinic settings.
Evaluation

Submissions are evaluated by the average area under the ROC curve between the predicted confidence scores and the observed targets across the twelve targets:

The final score is, in other words, the macro-averaged AUC ROC.
Submission File

For each row in the test set, you must predict a confidence score for each of the twelve target labels. The file should contain a header and have the following format:

StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture
<uid_1>,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5
<uid_2>,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5
...

Timeline

    July 30, 2026 - Start Date.
    October 15, 2026 - Entry Deadline. You must accept the competition rules before this date in order to compete.
    October 15, 2026 - Team Merger Deadline. This is the last day participants may join or merge teams.
    October 22, 2026 - Final Submission Deadline.
    November 5, 2026 - Winners' Requirement Deadline. This is the deadline for winners to submit to the host/Kaggle their training code, video and method description.

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.
Prizes

Main Leaderboard
First Prize: $9,000
Second Prize: $7,000
Third Prize: $6,500
Fourth Prize: $6,000
Fifth Prize: $5,500
Sixth Prize: $5,000
Seventh Prize: $5,000
Eighth Prize: $5,000
Ninth Prize: $5,000
Tenth Prize: $5,000

Efficiency Track
First Efficiency Prize: $7,000
Second Efficiency Prize: $6,000
Third Efficiency Prize: $5,000

Because this competition is being hosted in coordination with the Radiological Society of North America (RSNA) Annual Meeting, winners will be invited and strongly encouraged to attend the AI Challenge Recognition Event with waived fee, contingent on review of solution and fulfillment of winners' obligations.

Note that, per the competition rules, in addition to the standard Kaggle Winners' Obligations (open-source licensing requirements, solution packaging/delivery, presentation to host), the host team also asks that you:

(i) create a short video presenting your approach and solution, and

(ii) publish a link to your open sourced code and the weights on the competition forum

(iii) Share final version of model as publicly available for open distribution and validation. Please see https://www.kaggle.com/models/tom99763/9th-place-models-rsna-iad/PyTorch/default as an example.
Code Requirements

Submissions to this competition must be made through Notebooks. In order for the "Submit" button to be active after a commit, the following conditions must be met:

    CPU Notebook <= 9 hours run-time
    GPU Notebook <= 9 hours run-time
    Internet access disabled
    Freely & publicly available external data is allowed, including pre-trained models
    Submission file must be named submission.csv

Please see the Code Competition FAQ for more information on how to submit. And review the code debugging doc if you are encountering submission errors.
Efficiency Prize Evaluation
Efficiency Prize

We are hosting a second track that focuses on model efficiency, because highly accurate models are often computationally heavy.

For the Efficiency Prize, we will evaluate submissions on both runtime and predictive performance.

To be eligible for an Efficiency Prize, a submission:

    Must be among the submissions selected by a team for the Leaderboard Prize, or else among those submissions automatically selected under the conditions described in the My Submissions tab.
    Must be ranked on the Private Leaderboard higher than the sample_submission.csv benchmark.

All submissions meeting these conditions will be considered for the Efficiency Prize. A submission may be eligible for both the Leaderboard Prize and the Efficiency Prize.

An Efficiency Prize will be awarded to eligible submissions according to how they are ranked by the following evaluation metric on the private test data. See the Prizes tab for the prize awarded to each rank. More details may be posted via discussion forum updates.
Efficiency Score

We compute a submission's efficiency score by:

where is the submission's score on the main competition metric, is the score of the benchmark sample_submission.csv, is the maximum of all submissions on the Private Leaderboard, and is the number of seconds it takes for the submission to be evaluated. The objective is to minimize the efficiency score.

During the training period of the competition, you may see a leaderboard for the public test data in the following notebook, updated daily: Efficiency Leaderboard. After the competition ends, we will update this leaderboard with efficiency scores on the private data. During the training period, this leaderboard will show only the rank of each team, but not the complete score.
Citation

Po-Hao “Howard” Chen, Naveen Subhas, Robyn Ball, Pieter Baeyens, Errol Colak, Ali Emami, Hillary Garner, Jacob Kazam, Hui-Ming Lin, Luciano Prevedello, Daniel Schneider, Jason Sho, Ryan Holbrook, and María Cruz. RSNA Knee Abnormality Detection. https://kaggle.com/competitions/rsna-knee-abnormality-detection, 2026. Kaggle.

Dataset Description
Dataset Description

This dataset contains knee MRI studies annotated for twelve common findings: ligament and meniscus injuries, three compartments of osteoarthritis, joint effusion, synovitis, Baker's cyst, bone contusion, and fracture. Each study comprises a collection of individual MRI sequences from a single scanning session formatted as DICOM series. Your task is to predict the per-study probability of each of the twelve findings.

Studies come from a diverse international mix of imaging sites and span a wide range of scanners, protocols, and populations. Only a small subset of training studies carry per-condition labels. We also provide the original text of the radiology report from which you may wish to derive the labels for the remaining studies.
Files

train.csv One row per training study.

    StudyInstanceUID - unique identifier for the study; matches the folder name under train_series/.
    PatientSex - patient sex (Male or Female; may be blank).
    Report - the free-text radiology report. May be in any of several languages, depending on the reporting institution.

Twelve binary labels:

    ACL - anterior cruciate ligament injury (0/1).
    MCL - medial collateral ligament injury (0/1).
    Medial Meniscus - medial meniscus tear (0/1).
    Lateral Meniscus - lateral meniscus tear (0/1).
    Medial OA - osteoarthritis of the medial tibiofemoral compartment (0/1).
    Lateral OA - osteoarthritis of the lateral tibiofemoral compartment (0/1).
    PF OA - patellofemoral osteoarthritis (0/1).
    Effusion - joint effusion / excess fluid (0/1).
    Synovitis - inflammation of the joint lining (0/1).
    Baker's - Baker's cyst (0/1).
    Contusion - bone contusion / bone bruise (0/1).
    Fracture - fracture (0/1).

train_series.csv One row per training series. Each series is a single MRI acquisition and each study comprises several series.

    StudyInstanceUID - study this series belongs to.
    SeriesInstanceUID - unique identifier for the series; matches the folder name under train_series/<StudyInstanceUID>/.
    Fluid_Sensitive - 1 if the sequence emphasizes fluid signal (T2, PD, STIR, and similar), 0 otherwise.
    Fat_Suppression - 1 if the sequence applies fat suppression, 0 otherwise.
    Anatomical_Plane - imaging plane: Sagittal, Coronal, or Axial.

train_series/ Training DICOMs, organized as train_series/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm. Each .dcm is a single image slice. Series typically contain 20–45 slices (median 30), with a long tail out to a few hundred.

test.csv Example test file with three study IDs from the public test set. During scoring, this example data will be replaced with the actual test data. There are about 1300 studies in the test set.

    StudyInstanceUID - unique identifier for a test study.

test_series.csv Same schema as train_series.csv, for the example test studies. Replaced with the real test-series descriptors during scoring.

test_series/ Example test DICOMs, same layout as train_series/. Replaced with the real test DICOMs during scoring.

sample_submission.csv A valid submission with all label columns set to 0.5.
Dataset Distribution Notice

Although efforts have been made to ensure each abnormality is represented in each dataset, the prevalence of abnormalities is not guaranteed to be the same across the training, public leaderboard, and final evaluation datasets.
DICOM Notes

Intensities, orientations, and resolutions vary across series and studies. Series come in a mix of transfer syntaxes (uncompressed Explicit VR Little Endian, JPEG Lossless, JPEG 2000, Implicit VR Little Endian). Every DICOM has been stripped to an allowlisted set of 86 metadata tags.
Data Explorer

569.76 GB

    test_series

    train_series

    sample_submission.csv

    test.csv

    test_series.csv

    train.csv

    train_series.csv
    
    
    some examples attached, ipynb which are good ranked : 
    
    
RSNA Knee baseline v1
Updated 5h ago Score: 0.824 · 6 comments · RSNA Knee LLM-read report labels +3
134
Silver
RSNA Knee DINOv2 at meniscus resolution
Updated 20h ago Score: 0.815 · 0 comments · [Private Datasource] +2
RSNA Knee Public 4-fold DINOv2 v4
Updated 13h ago Score: 0.806 · 0 comments · RSNA Knee Abnormality Detection +2
20
rsna26-knee-eda
Updated 7h ago 0 comments · RSNA Knee Abnormality Detection
4
🧠🔬 RSNA Knee — Plane-Aware 2.5D MIL GPU Baseline
Updated 5h ago Score: 0.513 · 0 comments · RSNA Knee Abnormality Detection
2
RSNA | Metadata Probe
Updated 5h ago Score: 0.556 · 0 comments · RSNA Knee Abnormality Detection

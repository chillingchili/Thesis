window.RESEARCH_DATA = {
  "snapshot": "2026-10-01",
  "rows": [
    {
      "id": "legacy_gru",
      "group": "mobile",
      "name": "Original GRU ensemble",
      "mode": "all",
      "detail": "Bundled Android assets. First 128 valid frames, fresh Lite poses. Recorded-window evaluation; no physical phone test.",
      "diagnostic": {
        "accuracy": 0.5142857142857142,
        "macro_f1": 0.47891806574441304,
        "n": 175,
        "correct": 90,
        "confusion": [
          [
            6,
            4,
            51,
            0
          ],
          [
            12,
            35,
            11,
            0
          ],
          [
            5,
            2,
            49,
            0
          ]
        ],
        "recall": [
          0.09836065573770492,
          0.603448275862069,
          0.875
        ],
        "precision": [
          0.2608695652173913,
          0.8536585365853658,
          0.44144144144144143
        ]
      }
    },
    {
      "id": "legacy_knn",
      "group": "mobile",
      "name": "kNN5 \u00b7 timing features",
      "mode": "all",
      "detail": "Bundled Android assets. First 128 valid frames, fresh Lite poses. Recorded-window evaluation; no physical phone test.",
      "diagnostic": {
        "accuracy": 0.45714285714285713,
        "macro_f1": 0.4128212688584206,
        "n": 175,
        "correct": 80,
        "confusion": [
          [
            44,
            2,
            15,
            0
          ],
          [
            37,
            7,
            14,
            0
          ],
          [
            26,
            1,
            29,
            0
          ]
        ],
        "recall": [
          0.7213114754098361,
          0.1206896551724138,
          0.5178571428571429
        ],
        "precision": [
          0.411214953271028,
          0.7,
          0.5
        ]
      }
    },
    {
      "id": "legacy_hybrid",
      "group": "mobile",
      "name": "Original GRU + kNN5",
      "mode": "all",
      "detail": "Bundled Android assets. First 128 valid frames, fresh Lite poses. Recorded-window evaluation; no physical phone test.",
      "diagnostic": {
        "accuracy": 0.49714285714285716,
        "macro_f1": 0.49576667216449505,
        "n": 175,
        "correct": 87,
        "confusion": [
          [
            28,
            1,
            32,
            0
          ],
          [
            24,
            20,
            14,
            0
          ],
          [
            16,
            1,
            39,
            0
          ]
        ],
        "recall": [
          0.45901639344262296,
          0.3448275862068966,
          0.6964285714285714
        ],
        "precision": [
          0.4117647058823529,
          0.9090909090909091,
          0.4588235294117647
        ]
      }
    },
    {
      "id": "v2_ensemble",
      "group": "v2",
      "name": "Retrained GRU \u00b7 five-fold ensemble",
      "mode": "gru",
      "diagnostic": {
        "accuracy": 0.4685714285714286,
        "n": 175,
        "correct": 83
      },
      "detail": "209 eligible annotated training clips. Complete-serve resampling and feature augmentation. Diagnostic quality failures count as incorrect.",
      "stratified": {
        "accuracy": 0.9090909090909091,
        "macro_f1": 0.9092870951197382,
        "n": 209,
        "correct": 190
      },
      "player": {
        "accuracy": 0.36363636363636365,
        "macro_f1": 0.3659022340681511,
        "n": 209,
        "correct": 76
      }
    },
    {
      "id": "v2_single",
      "group": "v2",
      "name": "Retrained GRU \u00b7 single fit",
      "mode": "gru",
      "diagnostic": {
        "accuracy": 0.33714285714285713,
        "n": 175,
        "correct": 59
      },
      "detail": "209 eligible annotated training clips. Complete-serve resampling and feature augmentation. Diagnostic quality failures count as incorrect."
    },
    {
      "id": "v2_pilot",
      "group": "v2",
      "name": "Retraining pilot \u00b7 inner early stopping",
      "mode": "gru",
      "stratified": {
        "accuracy": 0.7368421052631579,
        "macro_f1": 0.733827057886483,
        "n": 209,
        "correct": 154
      },
      "player": {
        "accuracy": 0.33014354066985646,
        "macro_f1": 0.34883696425278576,
        "n": 209,
        "correct": 69
      },
      "detail": "Training-only pilot. Varied stopping epochs; no diagnostic score is reported here."
    },
    {
      "id": "v3_angles_control_gru",
      "group": "v3",
      "name": "Five angles \u00b7 control",
      "mode": "gru",
      "stratified": {
        "accuracy": 0.8755980861244019,
        "macro_f1": 0.8760053444165884,
        "n": 209,
        "correct": 183,
        "confusion": [
          [
            63,
            0,
            6,
            0
          ],
          [
            3,
            61,
            6,
            0
          ],
          [
            2,
            9,
            59,
            0
          ]
        ],
        "recall": [
          0.9130434782608695,
          0.8714285714285714,
          0.8428571428571429
        ],
        "precision": [
          0.9264705882352942,
          0.8714285714285714,
          0.8309859154929577
        ]
      },
      "player": {
        "accuracy": 0.36363636363636365,
        "macro_f1": 0.3152268238275489,
        "n": 209,
        "correct": 76,
        "confusion": [
          [
            16,
            3,
            50,
            0
          ],
          [
            30,
            8,
            32,
            0
          ],
          [
            12,
            6,
            52,
            0
          ]
        ],
        "recall": [
          0.2318840579710145,
          0.11428571428571428,
          0.7428571428571429
        ],
        "precision": [
          0.27586206896551724,
          0.47058823529411764,
          0.3880597014925373
        ]
      },
      "diagnostic": {
        "accuracy": 0.46285714285714286,
        "macro_f1": 0.43686703863564585,
        "n": 175,
        "correct": 81,
        "confusion": [
          [
            42,
            1,
            17,
            1
          ],
          [
            30,
            11,
            17,
            0
          ],
          [
            28,
            0,
            28,
            0
          ]
        ],
        "recall": [
          0.6885245901639344,
          0.1896551724137931,
          0.5
        ],
        "precision": [
          0.42,
          0.9166666666666666,
          0.45161290322580644
        ]
      },
      "detail": "Matched 100-epoch experiments. Diagnostic GRU averages five stratified-fold models; research hybrid uses a training-only 209-clip timing bank. Richer inputs also change model capacity."
    },
    {
      "id": "v3_angles_control_hybrid",
      "group": "v3",
      "name": "Five angles \u00b7 control",
      "mode": "hybrid",
      "stratified": {
        "accuracy": 0.8755980861244019,
        "macro_f1": 0.8761130055887089,
        "n": 209,
        "correct": 183,
        "confusion": [
          [
            61,
            0,
            8,
            0
          ],
          [
            4,
            61,
            5,
            0
          ],
          [
            2,
            7,
            61,
            0
          ]
        ],
        "recall": [
          0.8840579710144928,
          0.8714285714285714,
          0.8714285714285714
        ],
        "precision": [
          0.9104477611940298,
          0.8970588235294118,
          0.8243243243243243
        ]
      },
      "player": {
        "accuracy": 0.5311004784688995,
        "macro_f1": 0.5281032304911163,
        "n": 209,
        "correct": 111,
        "confusion": [
          [
            30,
            0,
            39,
            0
          ],
          [
            8,
            27,
            35,
            0
          ],
          [
            12,
            4,
            54,
            0
          ]
        ],
        "recall": [
          0.43478260869565216,
          0.38571428571428573,
          0.7714285714285715
        ],
        "precision": [
          0.6,
          0.8709677419354839,
          0.421875
        ]
      },
      "diagnostic": {
        "accuracy": 0.44571428571428573,
        "macro_f1": 0.4004001600640256,
        "n": 175,
        "correct": 78,
        "confusion": [
          [
            52,
            0,
            8,
            1
          ],
          [
            43,
            10,
            5,
            0
          ],
          [
            40,
            0,
            16,
            0
          ]
        ],
        "recall": [
          0.8524590163934426,
          0.1724137931034483,
          0.2857142857142857
        ],
        "precision": [
          0.3851851851851852,
          1.0,
          0.5517241379310345
        ]
      },
      "detail": "Matched 100-epoch experiments. Diagnostic GRU averages five stratified-fold models; research hybrid uses a training-only 209-clip timing bank. Richer inputs also change model capacity."
    },
    {
      "id": "v3_angles_thesis_aug_gru",
      "group": "v3",
      "name": "Five angles + video augmentation",
      "mode": "gru",
      "stratified": {
        "accuracy": 0.8564593301435407,
        "macro_f1": 0.8571465701290263,
        "n": 209,
        "correct": 179,
        "confusion": [
          [
            58,
            1,
            10,
            0
          ],
          [
            5,
            58,
            7,
            0
          ],
          [
            3,
            4,
            63,
            0
          ]
        ],
        "recall": [
          0.8405797101449275,
          0.8285714285714286,
          0.9
        ],
        "precision": [
          0.8787878787878788,
          0.9206349206349206,
          0.7875
        ]
      },
      "player": {
        "accuracy": 0.4784688995215311,
        "macro_f1": 0.4763180091936767,
        "n": 209,
        "correct": 100,
        "confusion": [
          [
            20,
            3,
            46,
            0
          ],
          [
            9,
            34,
            27,
            0
          ],
          [
            16,
            8,
            46,
            0
          ]
        ],
        "recall": [
          0.2898550724637681,
          0.4857142857142857,
          0.6571428571428571
        ],
        "precision": [
          0.4444444444444444,
          0.7555555555555555,
          0.3865546218487395
        ]
      },
      "diagnostic": {
        "accuracy": 0.4742857142857143,
        "macro_f1": 0.4226199301505716,
        "n": 175,
        "correct": 83,
        "confusion": [
          [
            41,
            0,
            19,
            1
          ],
          [
            29,
            6,
            23,
            0
          ],
          [
            20,
            0,
            36,
            0
          ]
        ],
        "recall": [
          0.6721311475409836,
          0.10344827586206896,
          0.6428571428571429
        ],
        "precision": [
          0.45555555555555555,
          1.0,
          0.46153846153846156
        ]
      },
      "detail": "Matched 100-epoch experiments. Diagnostic GRU averages five stratified-fold models; research hybrid uses a training-only 209-clip timing bank. Richer inputs also change model capacity."
    },
    {
      "id": "v3_angles_thesis_aug_hybrid",
      "group": "v3",
      "name": "Five angles + video augmentation",
      "mode": "hybrid",
      "stratified": {
        "accuracy": 0.861244019138756,
        "macro_f1": 0.861902906279806,
        "n": 209,
        "correct": 180,
        "confusion": [
          [
            61,
            0,
            8,
            0
          ],
          [
            6,
            57,
            7,
            0
          ],
          [
            5,
            3,
            62,
            0
          ]
        ],
        "recall": [
          0.8840579710144928,
          0.8142857142857143,
          0.8857142857142857
        ],
        "precision": [
          0.8472222222222222,
          0.95,
          0.8051948051948052
        ]
      },
      "player": {
        "accuracy": 0.5789473684210527,
        "macro_f1": 0.5859238054335196,
        "n": 209,
        "correct": 121,
        "confusion": [
          [
            30,
            1,
            38,
            0
          ],
          [
            2,
            39,
            29,
            0
          ],
          [
            15,
            3,
            52,
            0
          ]
        ],
        "recall": [
          0.43478260869565216,
          0.5571428571428572,
          0.7428571428571429
        ],
        "precision": [
          0.6382978723404256,
          0.9069767441860465,
          0.4369747899159664
        ]
      },
      "diagnostic": {
        "accuracy": 0.4342857142857143,
        "macro_f1": 0.38892056638765365,
        "n": 175,
        "correct": 76,
        "confusion": [
          [
            51,
            0,
            9,
            1
          ],
          [
            47,
            9,
            2,
            0
          ],
          [
            40,
            0,
            16,
            0
          ]
        ],
        "recall": [
          0.8360655737704918,
          0.15517241379310345,
          0.2857142857142857
        ],
        "precision": [
          0.3695652173913043,
          1.0,
          0.5925925925925926
        ]
      },
      "detail": "Matched 100-epoch experiments. Diagnostic GRU averages five stratified-fold models; research hybrid uses a training-only 209-clip timing bank. Richer inputs also change model capacity."
    },
    {
      "id": "v3_skeleton_gru",
      "group": "v3",
      "name": "Full skeleton",
      "mode": "gru",
      "stratified": {
        "accuracy": 0.9186602870813397,
        "macro_f1": 0.9187273977769396,
        "n": 209,
        "correct": 192,
        "confusion": [
          [
            63,
            1,
            5,
            0
          ],
          [
            1,
            66,
            3,
            0
          ],
          [
            4,
            3,
            63,
            0
          ]
        ],
        "recall": [
          0.9130434782608695,
          0.9428571428571428,
          0.9
        ],
        "precision": [
          0.9264705882352942,
          0.9428571428571428,
          0.8873239436619719
        ]
      },
      "player": {
        "accuracy": 0.35406698564593303,
        "macro_f1": 0.3204027601125023,
        "n": 209,
        "correct": 74,
        "confusion": [
          [
            24,
            11,
            34,
            0
          ],
          [
            23,
            6,
            41,
            0
          ],
          [
            11,
            15,
            44,
            0
          ]
        ],
        "recall": [
          0.34782608695652173,
          0.08571428571428572,
          0.6285714285714286
        ],
        "precision": [
          0.41379310344827586,
          0.1875,
          0.3697478991596639
        ]
      },
      "diagnostic": {
        "accuracy": 0.5542857142857143,
        "macro_f1": 0.4684888613457712,
        "n": 175,
        "correct": 97,
        "confusion": [
          [
            54,
            6,
            0,
            1
          ],
          [
            13,
            41,
            4,
            0
          ],
          [
            44,
            10,
            2,
            0
          ]
        ],
        "recall": [
          0.8852459016393442,
          0.7068965517241379,
          0.03571428571428571
        ],
        "precision": [
          0.4864864864864865,
          0.7192982456140351,
          0.3333333333333333
        ]
      },
      "detail": "Matched 100-epoch experiments. Diagnostic GRU averages five stratified-fold models; research hybrid uses a training-only 209-clip timing bank. Richer inputs also change model capacity."
    },
    {
      "id": "v3_skeleton_hybrid",
      "group": "v3",
      "name": "Full skeleton",
      "mode": "hybrid",
      "stratified": {
        "accuracy": 0.9186602870813397,
        "macro_f1": 0.9189359483504536,
        "n": 209,
        "correct": 192,
        "confusion": [
          [
            64,
            0,
            5,
            0
          ],
          [
            2,
            65,
            3,
            0
          ],
          [
            5,
            2,
            63,
            0
          ]
        ],
        "recall": [
          0.927536231884058,
          0.9285714285714286,
          0.9
        ],
        "precision": [
          0.9014084507042254,
          0.9701492537313433,
          0.8873239436619719
        ]
      },
      "player": {
        "accuracy": 0.5311004784688995,
        "macro_f1": 0.5323242794940909,
        "n": 209,
        "correct": 111,
        "confusion": [
          [
            41,
            0,
            28,
            0
          ],
          [
            6,
            25,
            39,
            0
          ],
          [
            14,
            11,
            45,
            0
          ]
        ],
        "recall": [
          0.5942028985507246,
          0.35714285714285715,
          0.6428571428571429
        ],
        "precision": [
          0.6721311475409836,
          0.6944444444444444,
          0.4017857142857143
        ]
      },
      "diagnostic": {
        "accuracy": 0.49142857142857144,
        "macro_f1": 0.44179169400325397,
        "n": 175,
        "correct": 86,
        "confusion": [
          [
            57,
            2,
            1,
            1
          ],
          [
            36,
            20,
            2,
            0
          ],
          [
            45,
            2,
            9,
            0
          ]
        ],
        "recall": [
          0.9344262295081968,
          0.3448275862068966,
          0.16071428571428573
        ],
        "precision": [
          0.41304347826086957,
          0.8333333333333334,
          0.75
        ]
      },
      "detail": "Matched 100-epoch experiments. Diagnostic GRU averages five stratified-fold models; research hybrid uses a training-only 209-clip timing bank. Richer inputs also change model capacity."
    },
    {
      "id": "v3_skeleton_motion_gru",
      "group": "v3",
      "name": "Full skeleton + motion",
      "mode": "gru",
      "stratified": {
        "accuracy": 0.9186602870813397,
        "macro_f1": 0.9189294601123872,
        "n": 209,
        "correct": 192,
        "confusion": [
          [
            63,
            1,
            5,
            0
          ],
          [
            2,
            65,
            3,
            0
          ],
          [
            5,
            1,
            64,
            0
          ]
        ],
        "recall": [
          0.9130434782608695,
          0.9285714285714286,
          0.9142857142857143
        ],
        "precision": [
          0.9,
          0.9701492537313433,
          0.8888888888888888
        ]
      },
      "player": {
        "accuracy": 0.32057416267942584,
        "macro_f1": 0.3010425061019933,
        "n": 209,
        "correct": 67,
        "confusion": [
          [
            25,
            18,
            26,
            0
          ],
          [
            23,
            6,
            41,
            0
          ],
          [
            14,
            20,
            36,
            0
          ]
        ],
        "recall": [
          0.36231884057971014,
          0.08571428571428572,
          0.5142857142857142
        ],
        "precision": [
          0.4032258064516129,
          0.13636363636363635,
          0.34951456310679613
        ]
      },
      "diagnostic": {
        "accuracy": 0.56,
        "macro_f1": 0.46633003344185336,
        "n": 175,
        "correct": 98,
        "confusion": [
          [
            54,
            6,
            0,
            1
          ],
          [
            12,
            43,
            3,
            0
          ],
          [
            46,
            9,
            1,
            0
          ]
        ],
        "recall": [
          0.8852459016393442,
          0.7413793103448276,
          0.017857142857142856
        ],
        "precision": [
          0.48214285714285715,
          0.7413793103448276,
          0.25
        ]
      },
      "detail": "Matched 100-epoch experiments. Diagnostic GRU averages five stratified-fold models; research hybrid uses a training-only 209-clip timing bank. Richer inputs also change model capacity."
    },
    {
      "id": "v3_skeleton_motion_hybrid",
      "group": "v3",
      "name": "Full skeleton + motion",
      "mode": "hybrid",
      "stratified": {
        "accuracy": 0.9186602870813397,
        "macro_f1": 0.9188711935169719,
        "n": 209,
        "correct": 192,
        "confusion": [
          [
            63,
            1,
            5,
            0
          ],
          [
            1,
            66,
            3,
            0
          ],
          [
            6,
            1,
            63,
            0
          ]
        ],
        "recall": [
          0.9130434782608695,
          0.9428571428571428,
          0.9
        ],
        "precision": [
          0.9,
          0.9705882352941176,
          0.8873239436619719
        ]
      },
      "player": {
        "accuracy": 0.5023923444976076,
        "macro_f1": 0.502714342249226,
        "n": 209,
        "correct": 105,
        "confusion": [
          [
            42,
            3,
            24,
            0
          ],
          [
            7,
            24,
            39,
            0
          ],
          [
            17,
            14,
            39,
            0
          ]
        ],
        "recall": [
          0.6086956521739131,
          0.34285714285714286,
          0.5571428571428572
        ],
        "precision": [
          0.6363636363636364,
          0.5853658536585366,
          0.38235294117647056
        ]
      },
      "diagnostic": {
        "accuracy": 0.46285714285714286,
        "macro_f1": 0.38273045049291077,
        "n": 175,
        "correct": 81,
        "confusion": [
          [
            58,
            2,
            0,
            1
          ],
          [
            37,
            20,
            1,
            0
          ],
          [
            51,
            2,
            3,
            0
          ]
        ],
        "recall": [
          0.9508196721311475,
          0.3448275862068966,
          0.05357142857142857
        ],
        "precision": [
          0.3972602739726027,
          0.8333333333333334,
          0.75
        ]
      },
      "detail": "Matched 100-epoch experiments. Diagnostic GRU averages five stratified-fold models; research hybrid uses a training-only 209-clip timing bank. Richer inputs also change model capacity."
    },
    {
      "id": "v4_angles_thesis_aug_gru",
      "group": "v4",
      "name": "Five angles + video augmentation",
      "mode": "gru",
      "player": {
        "accuracy": 0.4784688995215311,
        "macro_f1": 0.4763180091936767,
        "n": 209,
        "correct": 100,
        "confusion": [
          [
            20,
            3,
            46,
            0
          ],
          [
            9,
            34,
            27,
            0
          ],
          [
            16,
            8,
            46,
            0
          ]
        ],
        "recall": [
          0.2898550724637681,
          0.4857142857142857,
          0.6571428571428571
        ],
        "precision": [
          0.4444444444444444,
          0.7555555555555555,
          0.3865546218487395
        ],
        "class_recall": [
          0.2898550724637681,
          0.4857142857142857,
          0.6571428571428571
        ],
        "log_loss_available": 2.0092662305346605,
        "ece_available": 0.4386205099986501
      },
      "diagnostic": {
        "accuracy": 0.49714285714285716,
        "macro_f1": 0.47718137975004843,
        "n": 175,
        "correct": 87,
        "confusion": [
          [
            43,
            0,
            17,
            1
          ],
          [
            33,
            13,
            12,
            0
          ],
          [
            25,
            0,
            31,
            0
          ]
        ],
        "recall": [
          0.7049180327868853,
          0.22413793103448276,
          0.5535714285714286
        ],
        "precision": [
          0.42574257425742573,
          1.0,
          0.5166666666666667
        ],
        "class_recall": [
          0.7049180327868853,
          0.22413793103448276,
          0.5535714285714286
        ],
        "log_loss_available": 1.585554599761963,
        "ece_available": 0.3117257368633117
      },
      "selected": false,
      "detail": "Player CV uses nested training-only policy fitting for tuned fusion. Diagnostic results use one final full-training GRU per condition, unlike v3 ensembles. Selected on mean-player macro-F1; final coaching-confidence acceptance is disabled."
    },
    {
      "id": "v4_angles_thesis_aug_hybrid",
      "group": "v4",
      "name": "Five angles + video augmentation",
      "mode": "hybrid",
      "player": {
        "accuracy": 0.5789473684210527,
        "macro_f1": 0.5859238054335196,
        "n": 209,
        "correct": 121,
        "confusion": [
          [
            30,
            1,
            38,
            0
          ],
          [
            2,
            39,
            29,
            0
          ],
          [
            15,
            3,
            52,
            0
          ]
        ],
        "recall": [
          0.43478260869565216,
          0.5571428571428572,
          0.7428571428571429
        ],
        "precision": [
          0.6382978723404256,
          0.9069767441860465,
          0.4369747899159664
        ],
        "class_recall": [
          0.43478260869565216,
          0.5571428571428572,
          0.7428571428571429
        ],
        "log_loss_available": 1.429500911502629,
        "ece_available": 0.1620896064928924
      },
      "diagnostic": {
        "accuracy": 0.46285714285714286,
        "macro_f1": 0.42562942421367583,
        "n": 175,
        "correct": 81,
        "confusion": [
          [
            52,
            0,
            8,
            1
          ],
          [
            42,
            12,
            4,
            0
          ],
          [
            38,
            1,
            17,
            0
          ]
        ],
        "recall": [
          0.8524590163934426,
          0.20689655172413793,
          0.30357142857142855
        ],
        "precision": [
          0.3939393939393939,
          0.9230769230769231,
          0.5862068965517241
        ],
        "class_recall": [
          0.8524590163934426,
          0.20689655172413793,
          0.30357142857142855
        ],
        "log_loss_available": 1.6002291026685618,
        "ece_available": 0.3017682654096132
      },
      "selected": false,
      "detail": "Player CV uses nested training-only policy fitting for tuned fusion. Diagnostic results use one final full-training GRU per condition, unlike v3 ensembles. Selected on mean-player macro-F1; final coaching-confidence acceptance is disabled."
    },
    {
      "id": "v4_angles_thesis_aug_tuned",
      "group": "v4",
      "name": "Five angles + video augmentation",
      "mode": "tuned",
      "player": {
        "accuracy": 0.5311004784688995,
        "macro_f1": 0.5336134453781513,
        "n": 209,
        "correct": 111,
        "confusion": [
          [
            40,
            0,
            29,
            0
          ],
          [
            10,
            27,
            33,
            0
          ],
          [
            21,
            5,
            44,
            0
          ]
        ],
        "recall": [
          0.5797101449275363,
          0.38571428571428573,
          0.6285714285714286
        ],
        "precision": [
          0.5633802816901409,
          0.84375,
          0.41509433962264153
        ],
        "class_recall": [
          0.5797101449275363,
          0.38571428571428573,
          0.6285714285714286
        ],
        "log_loss_available": 1.0925670583557574,
        "ece_available": 0.11418617238451104
      },
      "diagnostic": {
        "accuracy": 0.42857142857142855,
        "macro_f1": 0.37399202105084467,
        "n": 175,
        "correct": 75,
        "confusion": [
          [
            53,
            0,
            7,
            1
          ],
          [
            48,
            9,
            1,
            0
          ],
          [
            42,
            1,
            13,
            0
          ]
        ],
        "recall": [
          0.8688524590163934,
          0.15517241379310345,
          0.23214285714285715
        ],
        "precision": [
          0.3706293706293706,
          0.9,
          0.6190476190476191
        ],
        "class_recall": [
          0.8688524590163934,
          0.15517241379310345,
          0.23214285714285715
        ],
        "log_loss_available": 1.0417317024678823,
        "ece_available": 0.07903524631402402
      },
      "selected": false,
      "detail": "Player CV uses nested training-only policy fitting for tuned fusion. Diagnostic results use one final full-training GRU per condition, unlike v3 ensembles. Selected on mean-player macro-F1; final coaching-confidence acceptance is disabled."
    },
    {
      "id": "v4_lean_control_gru",
      "group": "v4",
      "name": "Lean arm / torso \u00b7 control",
      "mode": "gru",
      "player": {
        "accuracy": 0.3588516746411483,
        "macro_f1": 0.3096156983480927,
        "n": 209,
        "correct": 75,
        "confusion": [
          [
            19,
            14,
            36,
            0
          ],
          [
            37,
            5,
            28,
            0
          ],
          [
            17,
            2,
            51,
            0
          ]
        ],
        "recall": [
          0.2753623188405797,
          0.07142857142857142,
          0.7285714285714285
        ],
        "precision": [
          0.2602739726027397,
          0.23809523809523808,
          0.4434782608695652
        ],
        "class_recall": [
          0.2753623188405797,
          0.07142857142857142,
          0.7285714285714285
        ],
        "log_loss_available": 4.031856548363316,
        "ece_available": 0.6131966222416271
      },
      "diagnostic": {
        "accuracy": 0.45714285714285713,
        "macro_f1": 0.41830065359477125,
        "n": 175,
        "correct": 80,
        "confusion": [
          [
            48,
            4,
            8,
            1
          ],
          [
            21,
            24,
            13,
            0
          ],
          [
            35,
            13,
            8,
            0
          ]
        ],
        "recall": [
          0.7868852459016393,
          0.41379310344827586,
          0.14285714285714285
        ],
        "precision": [
          0.46153846153846156,
          0.5853658536585366,
          0.27586206896551724
        ],
        "class_recall": [
          0.7868852459016393,
          0.41379310344827586,
          0.14285714285714285
        ],
        "log_loss_available": 2.66011118888855,
        "ece_available": 0.45345260082990274
      },
      "selected": false,
      "detail": "Player CV uses nested training-only policy fitting for tuned fusion. Diagnostic results use one final full-training GRU per condition, unlike v3 ensembles. Selected on mean-player macro-F1; final coaching-confidence acceptance is disabled."
    },
    {
      "id": "v4_lean_control_hybrid",
      "group": "v4",
      "name": "Lean arm / torso \u00b7 control",
      "mode": "hybrid",
      "player": {
        "accuracy": 0.5358851674641149,
        "macro_f1": 0.5345427059712774,
        "n": 209,
        "correct": 112,
        "confusion": [
          [
            37,
            0,
            32,
            0
          ],
          [
            15,
            26,
            29,
            0
          ],
          [
            19,
            2,
            49,
            0
          ]
        ],
        "recall": [
          0.5362318840579711,
          0.37142857142857144,
          0.7
        ],
        "precision": [
          0.5211267605633803,
          0.9285714285714286,
          0.44545454545454544
        ],
        "class_recall": [
          0.5362318840579711,
          0.37142857142857144,
          0.7
        ],
        "log_loss_available": 2.2331479795723808,
        "ece_available": 0.33337193210508687
      },
      "diagnostic": {
        "accuracy": 0.4057142857142857,
        "macro_f1": 0.3277063911984546,
        "n": 175,
        "correct": 71,
        "confusion": [
          [
            54,
            1,
            5,
            1
          ],
          [
            38,
            13,
            7,
            0
          ],
          [
            43,
            9,
            4,
            0
          ]
        ],
        "recall": [
          0.8852459016393442,
          0.22413793103448276,
          0.07142857142857142
        ],
        "precision": [
          0.4,
          0.5652173913043478,
          0.25
        ],
        "class_recall": [
          0.8852459016393442,
          0.22413793103448276,
          0.07142857142857142
        ],
        "log_loss_available": 2.0653775824305445,
        "ece_available": 0.36396951678396705
      },
      "selected": false,
      "detail": "Player CV uses nested training-only policy fitting for tuned fusion. Diagnostic results use one final full-training GRU per condition, unlike v3 ensembles. Selected on mean-player macro-F1; final coaching-confidence acceptance is disabled."
    },
    {
      "id": "v4_lean_control_tuned",
      "group": "v4",
      "name": "Lean arm / torso \u00b7 control",
      "mode": "tuned",
      "player": {
        "accuracy": 0.569377990430622,
        "macro_f1": 0.5691927282913968,
        "n": 209,
        "correct": 119,
        "confusion": [
          [
            43,
            0,
            26,
            0
          ],
          [
            12,
            28,
            30,
            0
          ],
          [
            19,
            3,
            48,
            0
          ]
        ],
        "recall": [
          0.6231884057971014,
          0.4,
          0.6857142857142857
        ],
        "precision": [
          0.581081081081081,
          0.9032258064516129,
          0.46153846153846156
        ],
        "class_recall": [
          0.6231884057971014,
          0.4,
          0.6857142857142857
        ],
        "log_loss_available": 1.1001866230984563,
        "ece_available": 0.2635052564143187
      },
      "diagnostic": {
        "accuracy": 0.44,
        "macro_f1": 0.3774442144007361,
        "n": 175,
        "correct": 77,
        "confusion": [
          [
            56,
            0,
            4,
            1
          ],
          [
            47,
            10,
            1,
            0
          ],
          [
            43,
            2,
            11,
            0
          ]
        ],
        "recall": [
          0.9180327868852459,
          0.1724137931034483,
          0.19642857142857142
        ],
        "precision": [
          0.3835616438356164,
          0.8333333333333334,
          0.6875
        ],
        "class_recall": [
          0.9180327868852459,
          0.1724137931034483,
          0.19642857142857142
        ],
        "log_loss_available": 1.0301920911090718,
        "ece_available": 0.11989939580944427
      },
      "selected": true,
      "detail": "Player CV uses nested training-only policy fitting for tuned fusion. Diagnostic results use one final full-training GRU per condition, unlike v3 ensembles. Selected on mean-player macro-F1; final coaching-confidence acceptance is disabled."
    },
    {
      "id": "v4_lean_thesis_aug_gru",
      "group": "v4",
      "name": "Lean arm / torso + augmentation",
      "mode": "gru",
      "player": {
        "accuracy": 0.3827751196172249,
        "macro_f1": 0.33970423186109455,
        "n": 209,
        "correct": 80,
        "confusion": [
          [
            20,
            13,
            36,
            0
          ],
          [
            33,
            8,
            29,
            0
          ],
          [
            13,
            5,
            52,
            0
          ]
        ],
        "recall": [
          0.2898550724637681,
          0.11428571428571428,
          0.7428571428571429
        ],
        "precision": [
          0.30303030303030304,
          0.3076923076923077,
          0.4444444444444444
        ],
        "class_recall": [
          0.2898550724637681,
          0.11428571428571428,
          0.7428571428571429
        ],
        "log_loss_available": 3.6291760497501797,
        "ece_available": 0.5753207279449444
      },
      "diagnostic": {
        "accuracy": 0.45714285714285713,
        "macro_f1": 0.4514938758776201,
        "n": 175,
        "correct": 80,
        "confusion": [
          [
            40,
            1,
            19,
            1
          ],
          [
            23,
            18,
            17,
            0
          ],
          [
            32,
            2,
            22,
            0
          ]
        ],
        "recall": [
          0.6557377049180327,
          0.3103448275862069,
          0.39285714285714285
        ],
        "precision": [
          0.42105263157894735,
          0.8571428571428571,
          0.3793103448275862
        ],
        "class_recall": [
          0.6557377049180327,
          0.3103448275862069,
          0.39285714285714285
        ],
        "log_loss_available": 2.6212809085845947,
        "ece_available": 0.4687815311996416
      },
      "selected": false,
      "detail": "Player CV uses nested training-only policy fitting for tuned fusion. Diagnostic results use one final full-training GRU per condition, unlike v3 ensembles. Selected on mean-player macro-F1; final coaching-confidence acceptance is disabled."
    },
    {
      "id": "v4_lean_thesis_aug_hybrid",
      "group": "v4",
      "name": "Lean arm / torso + augmentation",
      "mode": "hybrid",
      "player": {
        "accuracy": 0.5454545454545454,
        "macro_f1": 0.5383555076111711,
        "n": 209,
        "correct": 114,
        "confusion": [
          [
            38,
            2,
            29,
            0
          ],
          [
            15,
            25,
            30,
            0
          ],
          [
            13,
            6,
            51,
            0
          ]
        ],
        "recall": [
          0.5507246376811594,
          0.35714285714285715,
          0.7285714285714285
        ],
        "precision": [
          0.5757575757575758,
          0.7575757575757576,
          0.4636363636363636
        ],
        "class_recall": [
          0.5507246376811594,
          0.35714285714285715,
          0.7285714285714285
        ],
        "log_loss_available": 2.1075768068207714,
        "ece_available": 0.3253523798294325
      },
      "diagnostic": {
        "accuracy": 0.41714285714285715,
        "macro_f1": 0.37203683365717666,
        "n": 175,
        "correct": 73,
        "confusion": [
          [
            49,
            1,
            10,
            1
          ],
          [
            39,
            10,
            9,
            0
          ],
          [
            41,
            1,
            14,
            0
          ]
        ],
        "recall": [
          0.8032786885245902,
          0.1724137931034483,
          0.25
        ],
        "precision": [
          0.3798449612403101,
          0.8333333333333334,
          0.42424242424242425
        ],
        "class_recall": [
          0.8032786885245902,
          0.1724137931034483,
          0.25
        ],
        "log_loss_available": 2.204141776292966,
        "ece_available": 0.3414417002965157
      },
      "selected": false,
      "detail": "Player CV uses nested training-only policy fitting for tuned fusion. Diagnostic results use one final full-training GRU per condition, unlike v3 ensembles. Selected on mean-player macro-F1; final coaching-confidence acceptance is disabled."
    },
    {
      "id": "v4_lean_thesis_aug_tuned",
      "group": "v4",
      "name": "Lean arm / torso + augmentation",
      "mode": "tuned",
      "player": {
        "accuracy": 0.5358851674641149,
        "macro_f1": 0.5375412266250195,
        "n": 209,
        "correct": 112,
        "confusion": [
          [
            41,
            0,
            28,
            0
          ],
          [
            11,
            27,
            32,
            0
          ],
          [
            21,
            5,
            44,
            0
          ]
        ],
        "recall": [
          0.5942028985507246,
          0.38571428571428573,
          0.6285714285714286
        ],
        "precision": [
          0.5616438356164384,
          0.84375,
          0.4230769230769231
        ],
        "class_recall": [
          0.5942028985507246,
          0.38571428571428573,
          0.6285714285714286
        ],
        "log_loss_available": 1.0104984150408984,
        "ece_available": 0.08887375877982857
      },
      "diagnostic": {
        "accuracy": 0.42857142857142855,
        "macro_f1": 0.3745570823610751,
        "n": 175,
        "correct": 75,
        "confusion": [
          [
            53,
            0,
            7,
            1
          ],
          [
            47,
            10,
            1,
            0
          ],
          [
            42,
            2,
            12,
            0
          ]
        ],
        "recall": [
          0.8688524590163934,
          0.1724137931034483,
          0.21428571428571427
        ],
        "precision": [
          0.3732394366197183,
          0.8333333333333334,
          0.6
        ],
        "class_recall": [
          0.8688524590163934,
          0.1724137931034483,
          0.21428571428571427
        ],
        "log_loss_available": 1.0244552847415564,
        "ece_available": 0.034020923480748165
      },
      "selected": false,
      "detail": "Player CV uses nested training-only policy fitting for tuned fusion. Diagnostic results use one final full-training GRU per condition, unlike v3 ensembles. Selected on mean-player macro-F1; final coaching-confidence acceptance is disabled."
    },
    {
      "id": "v4_skeleton_thesis_aug_gru",
      "group": "v4",
      "name": "Full skeleton + augmentation",
      "mode": "gru",
      "player": {
        "accuracy": 0.3349282296650718,
        "macro_f1": 0.28653740374037406,
        "n": 209,
        "correct": 70,
        "confusion": [
          [
            23,
            7,
            39,
            0
          ],
          [
            22,
            1,
            47,
            0
          ],
          [
            6,
            18,
            46,
            0
          ]
        ],
        "recall": [
          0.3333333333333333,
          0.014285714285714285,
          0.6571428571428571
        ],
        "precision": [
          0.45098039215686275,
          0.038461538461538464,
          0.3484848484848485
        ],
        "class_recall": [
          0.3333333333333333,
          0.014285714285714285,
          0.6571428571428571
        ],
        "log_loss_available": 3.9879524627579985,
        "ece_available": 0.6286614173622223
      },
      "diagnostic": {
        "accuracy": 0.5657142857142857,
        "macro_f1": 0.49384132588016083,
        "n": 175,
        "correct": 99,
        "confusion": [
          [
            57,
            2,
            1,
            1
          ],
          [
            13,
            39,
            6,
            0
          ],
          [
            49,
            4,
            3,
            0
          ]
        ],
        "recall": [
          0.9344262295081968,
          0.6724137931034483,
          0.05357142857142857
        ],
        "precision": [
          0.4789915966386555,
          0.8666666666666667,
          0.3
        ],
        "class_recall": [
          0.9344262295081968,
          0.6724137931034483,
          0.05357142857142857
        ],
        "log_loss_available": 2.546877384185791,
        "ece_available": 0.3916703832560572
      },
      "selected": false,
      "detail": "Player CV uses nested training-only policy fitting for tuned fusion. Diagnostic results use one final full-training GRU per condition, unlike v3 ensembles. Selected on mean-player macro-F1; final coaching-confidence acceptance is disabled."
    },
    {
      "id": "v4_skeleton_thesis_aug_hybrid",
      "group": "v4",
      "name": "Full skeleton + augmentation",
      "mode": "hybrid",
      "player": {
        "accuracy": 0.5263157894736842,
        "macro_f1": 0.5280556383507203,
        "n": 209,
        "correct": 110,
        "confusion": [
          [
            41,
            0,
            28,
            0
          ],
          [
            6,
            24,
            40,
            0
          ],
          [
            9,
            16,
            45,
            0
          ]
        ],
        "recall": [
          0.5942028985507246,
          0.34285714285714286,
          0.6428571428571429
        ],
        "precision": [
          0.7321428571428571,
          0.6,
          0.39823008849557523
        ],
        "class_recall": [
          0.5942028985507246,
          0.34285714285714286,
          0.6428571428571429
        ],
        "log_loss_available": 2.117768238648208,
        "ece_available": 0.34667742445756866
      },
      "diagnostic": {
        "accuracy": 0.4685714285714286,
        "macro_f1": 0.3908969210174029,
        "n": 175,
        "correct": 82,
        "confusion": [
          [
            58,
            1,
            1,
            1
          ],
          [
            34,
            21,
            3,
            0
          ],
          [
            50,
            3,
            3,
            0
          ]
        ],
        "recall": [
          0.9508196721311475,
          0.3620689655172414,
          0.05357142857142857
        ],
        "precision": [
          0.4084507042253521,
          0.84,
          0.42857142857142855
        ],
        "class_recall": [
          0.9508196721311475,
          0.3620689655172414,
          0.05357142857142857
        ],
        "log_loss_available": 2.300634821010806,
        "ece_available": 0.33257655992270446
      },
      "selected": false,
      "detail": "Player CV uses nested training-only policy fitting for tuned fusion. Diagnostic results use one final full-training GRU per condition, unlike v3 ensembles. Selected on mean-player macro-F1; final coaching-confidence acceptance is disabled."
    },
    {
      "id": "v4_skeleton_thesis_aug_tuned",
      "group": "v4",
      "name": "Full skeleton + augmentation",
      "mode": "tuned",
      "player": {
        "accuracy": 0.5550239234449761,
        "macro_f1": 0.557454338784262,
        "n": 209,
        "correct": 116,
        "confusion": [
          [
            41,
            0,
            28,
            0
          ],
          [
            5,
            27,
            38,
            0
          ],
          [
            17,
            5,
            48,
            0
          ]
        ],
        "recall": [
          0.5942028985507246,
          0.38571428571428573,
          0.6857142857142857
        ],
        "precision": [
          0.6507936507936508,
          0.84375,
          0.42105263157894735
        ],
        "class_recall": [
          0.5942028985507246,
          0.38571428571428573,
          0.6857142857142857
        ],
        "log_loss_available": 1.0344079251706637,
        "ece_available": 0.15878374732053066
      },
      "diagnostic": {
        "accuracy": 0.42857142857142855,
        "macro_f1": 0.34825819052310464,
        "n": 175,
        "correct": 75,
        "confusion": [
          [
            58,
            0,
            2,
            1
          ],
          [
            48,
            10,
            0,
            0
          ],
          [
            48,
            1,
            7,
            0
          ]
        ],
        "recall": [
          0.9508196721311475,
          0.1724137931034483,
          0.125
        ],
        "precision": [
          0.37662337662337664,
          0.9090909090909091,
          0.7777777777777778
        ],
        "class_recall": [
          0.9508196721311475,
          0.1724137931034483,
          0.125
        ],
        "log_loss_available": 0.9957522099646283,
        "ece_available": 0.10662131111540366
      },
      "selected": false,
      "detail": "Player CV uses nested training-only policy fitting for tuned fusion. Diagnostic results use one final full-training GRU per condition, unlike v3 ensembles. Selected on mean-player macro-F1; final coaching-confidence acceptance is disabled."
    },
    {
      "id": "v4_skeleton_motion_thesis_aug_gru",
      "group": "v4",
      "name": "Skeleton / motion + augmentation",
      "mode": "gru",
      "player": {
        "accuracy": 0.3253588516746411,
        "macro_f1": 0.29834442737668543,
        "n": 209,
        "correct": 68,
        "confusion": [
          [
            24,
            7,
            38,
            0
          ],
          [
            23,
            5,
            42,
            0
          ],
          [
            8,
            23,
            39,
            0
          ]
        ],
        "recall": [
          0.34782608695652173,
          0.07142857142857142,
          0.5571428571428572
        ],
        "precision": [
          0.43636363636363634,
          0.14285714285714285,
          0.3277310924369748
        ],
        "class_recall": [
          0.34782608695652173,
          0.07142857142857142,
          0.5571428571428572
        ],
        "log_loss_available": 3.443126373843572,
        "ece_available": 0.6030924042161001
      },
      "diagnostic": {
        "accuracy": 0.52,
        "macro_f1": 0.4335714031908468,
        "n": 175,
        "correct": 91,
        "confusion": [
          [
            55,
            4,
            1,
            1
          ],
          [
            13,
            36,
            9,
            0
          ],
          [
            48,
            8,
            0,
            0
          ]
        ],
        "recall": [
          0.9016393442622951,
          0.6206896551724138,
          0.0
        ],
        "precision": [
          0.47413793103448276,
          0.75,
          0.0
        ],
        "class_recall": [
          0.9016393442622951,
          0.6206896551724138,
          0.0
        ],
        "log_loss_available": 3.251732110977173,
        "ece_available": 0.45576166113217664
      },
      "selected": false,
      "detail": "Player CV uses nested training-only policy fitting for tuned fusion. Diagnostic results use one final full-training GRU per condition, unlike v3 ensembles. Selected on mean-player macro-F1; final coaching-confidence acceptance is disabled."
    },
    {
      "id": "v4_skeleton_motion_thesis_aug_hybrid",
      "group": "v4",
      "name": "Skeleton / motion + augmentation",
      "mode": "hybrid",
      "player": {
        "accuracy": 0.507177033492823,
        "macro_f1": 0.5114824518478456,
        "n": 209,
        "correct": 106,
        "confusion": [
          [
            42,
            0,
            27,
            0
          ],
          [
            6,
            24,
            40,
            0
          ],
          [
            10,
            20,
            40,
            0
          ]
        ],
        "recall": [
          0.6086956521739131,
          0.34285714285714286,
          0.5714285714285714
        ],
        "precision": [
          0.7241379310344828,
          0.5454545454545454,
          0.37383177570093457
        ],
        "class_recall": [
          0.6086956521739131,
          0.34285714285714286,
          0.5714285714285714
        ],
        "log_loss_available": 1.7928218652314698,
        "ece_available": 0.29427393940648133
      },
      "diagnostic": {
        "accuracy": 0.41714285714285715,
        "macro_f1": 0.31869918699186994,
        "n": 175,
        "correct": 73,
        "confusion": [
          [
            57,
            2,
            1,
            1
          ],
          [
            35,
            16,
            7,
            0
          ],
          [
            52,
            4,
            0,
            0
          ]
        ],
        "recall": [
          0.9344262295081968,
          0.27586206896551724,
          0.0
        ],
        "precision": [
          0.3958333333333333,
          0.7272727272727273,
          0.0
        ],
        "class_recall": [
          0.9344262295081968,
          0.27586206896551724,
          0.0
        ],
        "log_loss_available": 2.545048125625298,
        "ece_available": 0.37440873511798195
      },
      "selected": false,
      "detail": "Player CV uses nested training-only policy fitting for tuned fusion. Diagnostic results use one final full-training GRU per condition, unlike v3 ensembles. Selected on mean-player macro-F1; final coaching-confidence acceptance is disabled."
    },
    {
      "id": "v4_skeleton_motion_thesis_aug_tuned",
      "group": "v4",
      "name": "Skeleton / motion + augmentation",
      "mode": "tuned",
      "player": {
        "accuracy": 0.5358851674641149,
        "macro_f1": 0.5375412266250195,
        "n": 209,
        "correct": 112,
        "confusion": [
          [
            41,
            0,
            28,
            0
          ],
          [
            11,
            27,
            32,
            0
          ],
          [
            21,
            5,
            44,
            0
          ]
        ],
        "recall": [
          0.5942028985507246,
          0.38571428571428573,
          0.6285714285714286
        ],
        "precision": [
          0.5616438356164384,
          0.84375,
          0.4230769230769231
        ],
        "class_recall": [
          0.5942028985507246,
          0.38571428571428573,
          0.6285714285714286
        ],
        "log_loss_available": 1.0104984150408984,
        "ece_available": 0.08887375877982857
      },
      "diagnostic": {
        "accuracy": 0.4114285714285714,
        "macro_f1": 0.324627616747182,
        "n": 175,
        "correct": 72,
        "confusion": [
          [
            57,
            0,
            3,
            1
          ],
          [
            48,
            10,
            0,
            0
          ],
          [
            50,
            1,
            5,
            0
          ]
        ],
        "recall": [
          0.9344262295081968,
          0.1724137931034483,
          0.08928571428571429
        ],
        "precision": [
          0.36774193548387096,
          0.9090909090909091,
          0.625
        ],
        "class_recall": [
          0.9344262295081968,
          0.1724137931034483,
          0.08928571428571429
        ],
        "log_loss_available": 1.0336750952089324,
        "ece_available": 0.16921900533915643
      },
      "selected": false,
      "detail": "Player CV uses nested training-only policy fitting for tuned fusion. Diagnostic results use one final full-training GRU per condition, unlike v3 ensembles. Selected on mean-player macro-F1; final coaching-confidence acceptance is disabled."
    },
    {
      "id": "archive_0",
      "group": "archive",
      "name": "Gradient Boosting",
      "mode": "all",
      "stratified": {
        "accuracy": 0.929
      },
      "player": {
        "accuracy": 0.5710000000000001
      },
      "diagnostic": {
        "accuracy": 0.449
      },
      "detail": "Reported in docs/RESULTS_SUMMARY.md, 24 Sep 2026. Diagnostic column is the mean of Beginner 3 and 4 accuracies, not pooled clip accuracy. Cached features and older protocols; exploratory adaptation reused these diagnostic players. Rounded historical scores have not been rerun for this website."
    },
    {
      "id": "archive_1",
      "group": "archive",
      "name": "Random Forest",
      "mode": "all",
      "stratified": {
        "accuracy": 0.9179999999999999
      },
      "player": {
        "accuracy": 0.578
      },
      "diagnostic": {
        "accuracy": 0.489
      },
      "detail": "Reported in docs/RESULTS_SUMMARY.md, 24 Sep 2026. Diagnostic column is the mean of Beginner 3 and 4 accuracies, not pooled clip accuracy. Cached features and older protocols; exploratory adaptation reused these diagnostic players. Rounded historical scores have not been rerun for this website."
    },
    {
      "id": "archive_2",
      "group": "archive",
      "name": "SVM \u00b7 RBF, scaled",
      "mode": "all",
      "stratified": {
        "accuracy": 0.899
      },
      "player": {
        "accuracy": 0.56
      },
      "diagnostic": {
        "accuracy": 0.48100000000000004
      },
      "detail": "Reported in docs/RESULTS_SUMMARY.md, 24 Sep 2026. Diagnostic column is the mean of Beginner 3 and 4 accuracies, not pooled clip accuracy. Cached features and older protocols; exploratory adaptation reused these diagnostic players. Rounded historical scores have not been rerun for this website."
    },
    {
      "id": "archive_3",
      "group": "archive",
      "name": "kNN7 \u00b7 scaled, Manhattan",
      "mode": "all",
      "stratified": {
        "accuracy": 0.879
      },
      "diagnostic": {
        "accuracy": 0.48700000000000004
      },
      "detail": "Reported in docs/RESULTS_SUMMARY.md, 24 Sep 2026. Diagnostic column is the mean of Beginner 3 and 4 accuracies, not pooled clip accuracy. Cached features and older protocols; exploratory adaptation reused these diagnostic players. Rounded historical scores have not been rerun for this website."
    },
    {
      "id": "archive_4",
      "group": "archive",
      "name": "Original GRU \u00b7 cached Heavy",
      "mode": "all",
      "stratified": {
        "accuracy": 0.867
      },
      "player": {
        "accuracy": 0.568
      },
      "diagnostic": {
        "accuracy": 0.38
      },
      "detail": "Reported in docs/RESULTS_SUMMARY.md, 24 Sep 2026. Diagnostic column is the mean of Beginner 3 and 4 accuracies, not pooled clip accuracy. Cached features and older protocols; exploratory adaptation reused these diagnostic players. Rounded historical scores have not been rerun for this website. The group-CV value comes from an older group run, not the stratified ensemble checkpoints."
    },
    {
      "id": "archive_5",
      "group": "archive",
      "name": "kNN7 \u00b7 scaled, Euclidean",
      "mode": "all",
      "stratified": {
        "accuracy": 0.8540000000000001
      },
      "detail": "Reported in docs/RESULTS_SUMMARY.md, 24 Sep 2026. Diagnostic column is the mean of Beginner 3 and 4 accuracies, not pooled clip accuracy. Cached features and older protocols; exploratory adaptation reused these diagnostic players. Rounded historical scores have not been rerun for this website."
    },
    {
      "id": "archive_6",
      "group": "archive",
      "name": "kNN5 \u00b7 scaled",
      "mode": "all",
      "stratified": {
        "accuracy": 0.84
      },
      "detail": "Reported in docs/RESULTS_SUMMARY.md, 24 Sep 2026. Diagnostic column is the mean of Beginner 3 and 4 accuracies, not pooled clip accuracy. Cached features and older protocols; exploratory adaptation reused these diagnostic players. Rounded historical scores have not been rerun for this website."
    },
    {
      "id": "archive_7",
      "group": "archive",
      "name": "Logistic Regression",
      "mode": "all",
      "stratified": {
        "accuracy": 0.831
      },
      "player": {
        "accuracy": 0.537
      },
      "diagnostic": {
        "accuracy": 0.485
      },
      "detail": "Reported in docs/RESULTS_SUMMARY.md, 24 Sep 2026. Diagnostic column is the mean of Beginner 3 and 4 accuracies, not pooled clip accuracy. Cached features and older protocols; exploratory adaptation reused these diagnostic players. Rounded historical scores have not been rerun for this website."
    },
    {
      "id": "archive_8",
      "group": "archive",
      "name": "kNN5 family \u00b7 raw timing features",
      "mode": "all",
      "stratified": {
        "accuracy": 0.7829999999999999
      },
      "diagnostic": {
        "accuracy": 0.561
      },
      "detail": "Reported in docs/RESULTS_SUMMARY.md, 24 Sep 2026. Diagnostic column is the mean of Beginner 3 and 4 accuracies, not pooled clip accuracy. Cached features and older protocols; exploratory adaptation reused these diagnostic players. Rounded historical scores have not been rerun for this website. The raw kNN CV entry and distance-weighted diagnostic variant are family-level results, not a matched ablation."
    },
    {
      "id": "archive_9",
      "group": "archive",
      "name": "GRU \u00b7 strong style augmentation",
      "mode": "all",
      "player": {
        "accuracy": 0.5529999999999999
      },
      "diagnostic": {
        "accuracy": 0.415
      },
      "detail": "Reported in docs/RESULTS_SUMMARY.md, 24 Sep 2026. Diagnostic column is the mean of Beginner 3 and 4 accuracies, not pooled clip accuracy. Cached features and older protocols; exploratory adaptation reused these diagnostic players. Rounded historical scores have not been rerun for this website."
    },
    {
      "id": "archive_10",
      "group": "archive",
      "name": "GRU \u00b7 metric / center loss",
      "mode": "all",
      "diagnostic": {
        "accuracy": 0.52
      },
      "detail": "Reported in docs/RESULTS_SUMMARY.md, 24 Sep 2026. Diagnostic column is the mean of Beginner 3 and 4 accuracies, not pooled clip accuracy. Cached features and older protocols; exploratory adaptation reused these diagnostic players. Rounded historical scores have not been rerun for this website."
    },
    {
      "id": "archive_11",
      "group": "archive",
      "name": "Moment-match TTA + ensemble",
      "mode": "all",
      "diagnostic": {
        "accuracy": 0.583
      },
      "detail": "Reported in docs/RESULTS_SUMMARY.md, 24 Sep 2026. Diagnostic column is the mean of Beginner 3 and 4 accuracies, not pooled clip accuracy. Cached features and older protocols; exploratory adaptation reused these diagnostic players. Rounded historical scores have not been rerun for this website."
    },
    {
      "id": "archive_12",
      "group": "archive",
      "name": "SVM + moment TTA",
      "mode": "all",
      "diagnostic": {
        "accuracy": 0.5720000000000001
      },
      "detail": "Reported in docs/RESULTS_SUMMARY.md, 24 Sep 2026. Diagnostic column is the mean of Beginner 3 and 4 accuracies, not pooled clip accuracy. Cached features and older protocols; exploratory adaptation reused these diagnostic players. Rounded historical scores have not been rerun for this website."
    },
    {
      "id": "archive_13",
      "group": "archive",
      "name": "kNN7 + biomechanical features",
      "mode": "all",
      "diagnostic": {
        "accuracy": 0.551
      },
      "detail": "Reported in docs/RESULTS_SUMMARY.md, 24 Sep 2026. Diagnostic column is the mean of Beginner 3 and 4 accuracies, not pooled clip accuracy. Cached features and older protocols; exploratory adaptation reused these diagnostic players. Rounded historical scores have not been rerun for this website."
    },
    {
      "id": "archive_14",
      "group": "archive",
      "name": "CORAL \u00b7 kNN5 / SVM / RF",
      "mode": "all",
      "diagnostic": {
        "accuracy": 0.557
      },
      "detail": "Reported in docs/RESULTS_SUMMARY.md, 24 Sep 2026. Diagnostic column is the mean of Beginner 3 and 4 accuracies, not pooled clip accuracy. Cached features and older protocols; exploratory adaptation reused these diagnostic players. Rounded historical scores have not been rerun for this website."
    },
    {
      "id": "archive_15",
      "group": "archive",
      "name": "Weighted kNN5 / kNN7 / RF",
      "mode": "all",
      "diagnostic": {
        "accuracy": 0.5660000000000001
      },
      "detail": "Reported in docs/RESULTS_SUMMARY.md, 24 Sep 2026. Diagnostic column is the mean of Beginner 3 and 4 accuracies, not pooled clip accuracy. Cached features and older protocols; exploratory adaptation reused these diagnostic players. Rounded historical scores have not been rerun for this website."
    }
  ],
  "detectors": [
    {
      "name": "Ball detector",
      "task": "Box detection",
      "epochs": 100,
      "precision": 0.95793,
      "recall": 0.92082,
      "map50": 0.93788,
      "map5095": 0.50382,
      "source": "assets/sources/ball-detector.csv"
    },
    {
      "name": "Paddle detector \u00b7 base",
      "task": "Box detection",
      "epochs": 100,
      "precision": 0.85256,
      "recall": 0.77098,
      "map50": 0.80938,
      "map5095": 0.36692,
      "source": "assets/sources/paddle-detector-base.csv"
    },
    {
      "name": "Paddle detector \u00b7 fine-tuned",
      "task": "Box detection",
      "epochs": 60,
      "precision": 0.98509,
      "recall": 0.73684,
      "map50": 0.84729,
      "map5095": 0.54533,
      "source": "assets/sources/paddle-detector-fine-tuned.csv"
    },
    {
      "name": "Court landmarks \u00b7 base",
      "task": "Court keypoints",
      "epochs": 100,
      "precision": 0.99936,
      "recall": 1.0,
      "map50": 0.995,
      "map5095": 0.97737,
      "source": "assets/sources/court-landmarks-base.csv"
    },
    {
      "name": "Court landmarks \u00b7 fine-tuned",
      "task": "Court keypoints",
      "epochs": 60,
      "precision": 0.99922,
      "recall": 1.0,
      "map50": 0.995,
      "map5095": 0.98411,
      "source": "assets/sources/court-landmarks-fine-tuned.csv"
    }
  ],
  "samples": [
    {
      "id": "drive-01",
      "name": "Beginner 2 \u00b7 Drive 01",
      "truth": "Drive",
      "label_source": "Coach B annotation \u00b7 training clip",
      "tab_label": "Drive \u00b7 correct",
      "role": "correct",
      "video": "assets/samples/drive-01.mp4",
      "poster": "assets/samples/drive-01.webp",
      "outputs": [
        {
          "name": "Retrained v2 GRU \u00b7 ensemble",
          "report": {
            "video": {
              "fps": 59.94005994005994,
              "frames": 142,
              "duration": 2.3690333333333333,
              "width": 1920,
              "height": 1080
            },
            "range": {
              "start": 0,
              "end": 2.3690333333333333
            },
            "options": {
              "start": 0,
              "end": null,
              "model": "v2_ensemble",
              "moment_matching": false,
              "landing": "off"
            },
            "warnings": [],
            "serve": {
              "status": "ok",
              "label": "drive",
              "confidence": 0.8252912759780884,
              "probabilities": {
                "drive": 0.8252912759780884,
                "lob": 0.0044945161789655685,
                "topspin": 0.1702142059803009
              },
              "model": "ensemble",
              "model_version": "serve_v2",
              "model_manifest_sha256": "8e265b1146a4ec53554ce17e1028be1482ad9c58b308cafc97c7ee7a18a82fbd",
              "feedback_eligible": true,
              "valid_frames": 128,
              "window_frames": 128,
              "truncated_frames": 0,
              "moment_matching": false,
              "preprocessing": {
                "contract": "serve_sequence_v2",
                "source_frames": 142,
                "valid_source_frames": 142,
                "valid_source_fraction": 1.0,
                "valid_window_frames": 128,
                "valid_window_fraction": 1.0,
                "quality_accepted": true,
                "window_frames": 128,
                "source_start_frame": 0,
                "source_end_frame": 141,
                "truncated_frames": 0,
                "moment_matching": false
              }
            },
            "shift": {
              "status": "ok",
              "value": 0.09710743959732729,
              "threshold": 0.1542655138505721,
              "sufficient": false
            },
            "paddle": {
              "status": "ok",
              "angle": 70.01766204833984,
              "low": 13.410494542121889,
              "high": 165.73870525360107,
              "state": "OPTIMAL",
              "measurements": 8,
              "direction_validated": false
            },
            "landing": {
              "status": "disabled"
            },
            "pose_coverage": 1.0,
            "contact": {
              "frame": 84,
              "seconds": 1.4014,
              "method": "peak right-wrist speed proxy"
            },
            "feedback": {
              "status": "CORRECTIVE",
              "messages": [
                "Transfer your weight forward through the serve instead of relying only on your arm."
              ],
              "notifications": [],
              "rule_ids": [
                "C4_WEIGHT_TRANSFER"
              ],
              "version": "coach-b-v1",
              "validation_status": "pending_coach_c"
            },
            "seconds": 16.94,
            "assets": {
              "manifest.json": "a08fe5749fc2b30decc3a72872f94c30e40fe512f396311ba757c7f98bf236b7",
              "feedback_rules.json": "0c2c562f06344bbe133d9c33f606640139f594e5d1d9b5654eb30b46c37c5fb8",
              "shift_config.json": "c6bf2283b2ac04faa12fd8001648765817f8e6df6ff49419f35d72f1d718e996",
              "paddle_config.json": "db559b8669ae16fa8bbfec678729ff5874d3e62d8db9b1ae80196685f29631e7"
            },
            "files": [
              "contact.jpg",
              "keypoints.npy",
              "preview.mp4"
            ]
          },
          "download": "assets/sources/drive-01-saved-report.json"
        }
      ]
    },
    {
      "id": "lob-23",
      "name": "Beginner 1 \u00b7 Lob 23",
      "truth": "Lob",
      "label_source": "Coach B annotation \u00b7 training clip",
      "tab_label": "Lob \u00b7 correct",
      "role": "correct",
      "video": "assets/samples/lob-23.mp4",
      "poster": "assets/samples/lob-23.webp",
      "outputs": [
        {
          "name": "Original GRU \u00b7 ensemble",
          "report": {
            "video": {
              "fps": 59.94005994005994,
              "frames": 183,
              "duration": 3.0530500000000003,
              "width": 1920,
              "height": 1080
            },
            "range": {
              "start": 0.0,
              "end": 3.0530500000000003
            },
            "options": {
              "start": 0.0,
              "end": null,
              "model": "ensemble",
              "moment_matching": true,
              "landing": "off"
            },
            "warnings": [
              "The legacy GRU uses only the first 128 frames. Its window differs from the retrained complete-serve model."
            ],
            "serve": {
              "status": "ok",
              "label": "lob",
              "confidence": 0.9835962057113647,
              "probabilities": {
                "drive": 0.008791176602244377,
                "lob": 0.9835962057113647,
                "topspin": 0.007612636778503656
              },
              "model": "ensemble",
              "model_version": "legacy",
              "model_manifest_sha256": "a08fe5749fc2b30decc3a72872f94c30e40fe512f396311ba757c7f98bf236b7",
              "feedback_eligible": true,
              "valid_frames": 128,
              "window_frames": 128,
              "truncated_frames": 55,
              "moment_matching": true,
              "preprocessing": {},
              "components": {}
            },
            "shift": {
              "status": "ok",
              "value": 0.13795681047236436,
              "threshold": 0.1542655138505721,
              "sufficient": false
            },
            "paddle": {
              "status": "unavailable",
              "reason": "No usable paddle orientation within five frames of estimated contact."
            },
            "landing": {
              "status": "disabled"
            },
            "pose_coverage": 1.0,
            "contact": {
              "frame": 122,
              "seconds": 2.0353666666666665,
              "method": "peak right-wrist speed proxy"
            },
            "feedback": {
              "status": "CORRECTIVE",
              "messages": [
                "Transfer your weight forward through the serve instead of relying only on your arm."
              ],
              "notifications": [
                "Paddle orientation unavailable."
              ],
              "rule_ids": [
                "C4_WEIGHT_TRANSFER"
              ],
              "version": "coach-b-v1",
              "validation_status": "pending_coach_c"
            },
            "seconds": 42.97,
            "assets": {
              "manifest.json": "a08fe5749fc2b30decc3a72872f94c30e40fe512f396311ba757c7f98bf236b7",
              "feedback_rules.json": "0c2c562f06344bbe133d9c33f606640139f594e5d1d9b5654eb30b46c37c5fb8",
              "shift_config.json": "c6bf2283b2ac04faa12fd8001648765817f8e6df6ff49419f35d72f1d718e996",
              "paddle_config.json": "db559b8669ae16fa8bbfec678729ff5874d3e62d8db9b1ae80196685f29631e7"
            },
            "files": [
              "contact.jpg",
              "keypoints.npy",
              "preview.mp4"
            ],
            "primary_filename": "Beginner1_Lob_023.mp4",
            "sample_purpose": "Curated website illustration; annotated training clip, not a performance estimate."
          },
          "download": "assets/sources/lob-23-saved-report.json"
        }
      ]
    },
    {
      "id": "topspin-22",
      "name": "Beginner 1 \u00b7 Topspin 22",
      "truth": "Topspin",
      "label_source": "Coach B annotation \u00b7 training clip",
      "tab_label": "Topspin \u00b7 correct",
      "role": "correct",
      "video": "assets/samples/topspin-22.mp4",
      "poster": "assets/samples/topspin-22.webp",
      "outputs": [
        {
          "name": "Original GRU \u00b7 ensemble",
          "report": {
            "video": {
              "fps": 59.94005994005994,
              "frames": 169,
              "duration": 2.8194833333333333,
              "width": 1920,
              "height": 1080
            },
            "range": {
              "start": 0.0,
              "end": 2.8194833333333333
            },
            "options": {
              "start": 0.0,
              "end": null,
              "model": "ensemble",
              "moment_matching": true,
              "landing": "off"
            },
            "warnings": [
              "The legacy GRU uses only the first 128 frames. Its window differs from the retrained complete-serve model."
            ],
            "serve": {
              "status": "ok",
              "label": "topspin",
              "confidence": 0.968558669090271,
              "probabilities": {
                "drive": 0.003677990986034274,
                "lob": 0.027763301506638527,
                "topspin": 0.968558669090271
              },
              "model": "ensemble",
              "model_version": "legacy",
              "model_manifest_sha256": "a08fe5749fc2b30decc3a72872f94c30e40fe512f396311ba757c7f98bf236b7",
              "feedback_eligible": true,
              "valid_frames": 128,
              "window_frames": 128,
              "truncated_frames": 41,
              "moment_matching": true,
              "preprocessing": {},
              "components": {}
            },
            "shift": {
              "status": "ok",
              "value": 0.14467596971577584,
              "threshold": 0.1542655138505721,
              "sufficient": false
            },
            "paddle": {
              "status": "unavailable",
              "reason": "No usable paddle orientation within five frames of estimated contact."
            },
            "landing": {
              "status": "disabled"
            },
            "pose_coverage": 1.0,
            "contact": {
              "frame": 110,
              "seconds": 1.8351666666666666,
              "method": "peak right-wrist speed proxy"
            },
            "feedback": {
              "status": "CORRECTIVE",
              "messages": [
                "Transfer your weight forward through the serve instead of relying only on your arm."
              ],
              "notifications": [
                "Paddle orientation unavailable."
              ],
              "rule_ids": [
                "C4_WEIGHT_TRANSFER"
              ],
              "version": "coach-b-v1",
              "validation_status": "pending_coach_c"
            },
            "seconds": 7.08,
            "assets": {
              "manifest.json": "a08fe5749fc2b30decc3a72872f94c30e40fe512f396311ba757c7f98bf236b7",
              "feedback_rules.json": "0c2c562f06344bbe133d9c33f606640139f594e5d1d9b5654eb30b46c37c5fb8",
              "shift_config.json": "c6bf2283b2ac04faa12fd8001648765817f8e6df6ff49419f35d72f1d718e996",
              "paddle_config.json": "db559b8669ae16fa8bbfec678729ff5874d3e62d8db9b1ae80196685f29631e7"
            },
            "files": [
              "keypoints.npy",
              "preview.mp4"
            ],
            "primary_filename": "Beginner1_Topspin_022.mp4",
            "sample_purpose": "Curated website illustration; annotated training clip, not a performance estimate."
          },
          "download": "assets/sources/topspin-22-saved-report.json"
        }
      ]
    },
    {
      "id": "drive-07",
      "name": "Beginner 2 \u00b7 Drive 07",
      "truth": "Drive",
      "label_source": "Coach B annotation \u00b7 training clip",
      "tab_label": "Error example",
      "role": "limitation",
      "video": "assets/samples/drive-07.mp4",
      "poster": "assets/samples/drive-07.webp",
      "outputs": [
        {
          "name": "Original GRU \u00b7 ensemble",
          "report": {
            "video": {
              "fps": 59.94005994005994,
              "frames": 143,
              "duration": 2.3857166666666667,
              "width": 1920,
              "height": 1080
            },
            "range": {
              "start": 0.0,
              "end": 2.3857166666666667
            },
            "options": {
              "start": 0.0,
              "end": null,
              "model": "ensemble",
              "landing": "off",
              "moment_matching": true
            },
            "warnings": [
              "The GRU uses the first 128 frames of the selected range, matching offline training. Trim to the serve motion if needed."
            ],
            "serve": {
              "status": "ok",
              "label": "lob",
              "confidence": 0.9650015830993652,
              "probabilities": {
                "drive": 0.008656986057758331,
                "lob": 0.9650015830993652,
                "topspin": 0.02634141966700554
              },
              "model": "ensemble",
              "valid_frames": 128,
              "window_frames": 128,
              "truncated_frames": 15,
              "moment_matching": true
            },
            "shift": {
              "status": "ok",
              "value": 0.07816709094943168,
              "threshold": 0.1542655138505721,
              "sufficient": false
            },
            "paddle": {
              "status": "ok",
              "angle": 78.11134338378906,
              "low": 13.410494542121889,
              "high": 165.73870525360107,
              "state": "OPTIMAL",
              "measurements": 9,
              "direction_validated": false
            },
            "landing": {
              "status": "disabled"
            },
            "pose_coverage": 1.0,
            "contact": {
              "frame": 83,
              "seconds": 1.3847166666666666,
              "method": "peak right-wrist speed proxy"
            },
            "feedback": {
              "status": "CORRECTIVE",
              "messages": [
                "Transfer your weight forward through the serve instead of relying only on your arm."
              ],
              "notifications": [],
              "rule_ids": [
                "C4_WEIGHT_TRANSFER"
              ],
              "version": "coach-b-v1",
              "validation_status": "pending_coach_c"
            },
            "seconds": 8.49,
            "assets": {
              "manifest.json": "a08fe5749fc2b30decc3a72872f94c30e40fe512f396311ba757c7f98bf236b7",
              "feedback_rules.json": "0c2c562f06344bbe133d9c33f606640139f594e5d1d9b5654eb30b46c37c5fb8",
              "shift_config.json": "c6bf2283b2ac04faa12fd8001648765817f8e6df6ff49419f35d72f1d718e996",
              "paddle_config.json": "db559b8669ae16fa8bbfec678729ff5874d3e62d8db9b1ae80196685f29631e7"
            },
            "files": [
              "contact.jpg",
              "keypoints.npy",
              "preview.mp4"
            ],
            "primary_filename": "Beginner2_Drive_007.mp4",
            "secondary_filename": null,
            "job_id": "cf549b29380f4ca7b55b01eb0a7030f4"
          },
          "download": "assets/sources/drive-07-saved-report.json"
        }
      ]
    }
  ],
  "calibration": [
    {
      "name": "Five angles + video augmentation",
      "before": 1.429500911502629,
      "after": 1.0925670583557574,
      "ece_before": 0.1620896064928924,
      "ece_after": 0.11418617238451104,
      "accepted_accuracy": 0.35
    },
    {
      "name": "Lean arm / torso \u00b7 control",
      "before": 2.2331479795723808,
      "after": 1.1001866230984563,
      "ece_before": 0.33337193210508687,
      "ece_after": 0.2635052564143187,
      "accepted_accuracy": 0.2564102564102564
    },
    {
      "name": "Lean arm / torso + augmentation",
      "before": 2.1075768068207714,
      "after": 1.0104984150408984,
      "ece_before": 0.3253523798294325,
      "ece_after": 0.08887375877982857,
      "accepted_accuracy": 0.2564102564102564
    },
    {
      "name": "Full skeleton + augmentation",
      "before": 2.117768238648208,
      "after": 1.0344079251706637,
      "ece_before": 0.34667742445756866,
      "ece_after": 0.15878374732053066,
      "accepted_accuracy": 0.2564102564102564
    },
    {
      "name": "Skeleton / motion + augmentation",
      "before": 1.7928218652314698,
      "after": 1.0104984150408984,
      "ece_before": 0.29427393940648133,
      "ece_after": 0.08887375877982857,
      "accepted_accuracy": 0.2564102564102564
    }
  ],
  "capture": {
    "first128": {
      "gru": {
        "n": 175,
        "correct": 90,
        "accuracy": 0.5142857142857142,
        "macro_f1": 0.47891806574441304,
        "confusion_with_unavailable": [
          [
            6,
            4,
            51,
            0
          ],
          [
            12,
            35,
            11,
            0
          ],
          [
            5,
            2,
            49,
            0
          ],
          [
            0,
            0,
            0,
            0
          ]
        ],
        "available": 175,
        "confidence_coverage": 0.7428571428571429,
        "confident_wrong": 63,
        "confident_accuracy": 0.5153846153846153,
        "per_subject": {
          "Beginner3": {
            "n": 85,
            "correct": 37,
            "accuracy": 0.43529411764705883,
            "macro_f1": 0.4289394985047159,
            "confusion_with_unavailable": [
              [
                6,
                4,
                21,
                0
              ],
              [
                12,
                12,
                4,
                0
              ],
              [
                5,
                2,
                19,
                0
              ],
              [
                0,
                0,
                0,
                0
              ]
            ],
            "available": 85,
            "confidence_coverage": 0.6470588235294118,
            "confident_wrong": 32,
            "confident_accuracy": 0.41818181818181815
          },
          "Beginner4": {
            "n": 90,
            "correct": 53,
            "accuracy": 0.5888888888888889,
            "macro_f1": 0.49549374311093825,
            "confusion_with_unavailable": [
              [
                0,
                0,
                30,
                0
              ],
              [
                0,
                23,
                7,
                0
              ],
              [
                0,
                0,
                30,
                0
              ],
              [
                0,
                0,
                0,
                0
              ]
            ],
            "available": 90,
            "confidence_coverage": 0.8333333333333334,
            "confident_wrong": 31,
            "confident_accuracy": 0.5866666666666667
          }
        },
        "mean_subject_accuracy": 0.5120915032679738,
        "log_loss_available": 1.2665324211120605,
        "brier_available": 0.7126183116784901,
        "reliability_bins": [
          {
            "low": 0.30000000000000004,
            "high": 0.4,
            "n": 2,
            "confidence": 0.35974735021591187,
            "accuracy": 0.5
          },
          {
            "low": 0.4,
            "high": 0.5,
            "n": 11,
            "confidence": 0.4724421203136444,
            "accuracy": 0.6363636363636364
          },
          {
            "low": 0.5,
            "high": 0.6000000000000001,
            "n": 32,
            "confidence": 0.5437455177307129,
            "accuracy": 0.46875
          },
          {
            "low": 0.6000000000000001,
            "high": 0.7000000000000001,
            "n": 30,
            "confidence": 0.6531116366386414,
            "accuracy": 0.5
          },
          {
            "low": 0.7000000000000001,
            "high": 0.8,
            "n": 33,
            "confidence": 0.7507960796356201,
            "accuracy": 0.42424242424242425
          },
          {
            "low": 0.8,
            "high": 0.9,
            "n": 33,
            "confidence": 0.8471822142601013,
            "accuracy": 0.45454545454545453
          },
          {
            "low": 0.9,
            "high": 1.0,
            "n": 34,
            "confidence": 0.9424440860748291,
            "accuracy": 0.6764705882352942
          }
        ],
        "ece_available": 0.23916131581578937,
        "selected_threshold": 0.6,
        "selected_coverage": 0.7428571428571429,
        "selected_accuracy": 0.5153846153846153,
        "selected_wrong": 63,
        "class_recall": [
          0.09836065573770492,
          0.603448275862069,
          0.875
        ]
      },
      "hybrid": {
        "n": 175,
        "correct": 87,
        "accuracy": 0.49714285714285716,
        "macro_f1": 0.49576667216449505,
        "confusion_with_unavailable": [
          [
            28,
            1,
            32,
            0
          ],
          [
            24,
            20,
            14,
            0
          ],
          [
            16,
            1,
            39,
            0
          ],
          [
            0,
            0,
            0,
            0
          ]
        ],
        "available": 175,
        "confidence_coverage": 0.5085714285714286,
        "confident_wrong": 41,
        "confident_accuracy": 0.5393258426966292,
        "per_subject": {
          "Beginner3": {
            "n": 85,
            "correct": 40,
            "accuracy": 0.47058823529411764,
            "macro_f1": 0.4421858334901813,
            "confusion_with_unavailable": [
              [
                15,
                1,
                15,
                0
              ],
              [
                18,
                5,
                5,
                0
              ],
              [
                5,
                1,
                20,
                0
              ],
              [
                0,
                0,
                0,
                0
              ]
            ],
            "available": 85,
            "confidence_coverage": 0.5882352941176471,
            "confident_wrong": 26,
            "confident_accuracy": 0.48
          },
          "Beginner4": {
            "n": 90,
            "correct": 47,
            "accuracy": 0.5222222222222223,
            "macro_f1": 0.5355555555555557,
            "confusion_with_unavailable": [
              [
                13,
                0,
                17,
                0
              ],
              [
                6,
                15,
                9,
                0
              ],
              [
                11,
                0,
                19,
                0
              ],
              [
                0,
                0,
                0,
                0
              ]
            ],
            "available": 90,
            "confidence_coverage": 0.43333333333333335,
            "confident_wrong": 15,
            "confident_accuracy": 0.6153846153846154
          }
        },
        "mean_subject_accuracy": 0.49640522875816995,
        "log_loss_available": 1.0844584703445435,
        "brier_available": 0.630811118959627,
        "reliability_bins": [
          {
            "low": 0.30000000000000004,
            "high": 0.4,
            "n": 2,
            "confidence": 0.3714832067489624,
            "accuracy": 1.0
          },
          {
            "low": 0.4,
            "high": 0.5,
            "n": 22,
            "confidence": 0.4575190246105194,
            "accuracy": 0.5454545454545454
          },
          {
            "low": 0.5,
            "high": 0.6000000000000001,
            "n": 62,
            "confidence": 0.5445479154586792,
            "accuracy": 0.4032258064516129
          },
          {
            "low": 0.6000000000000001,
            "high": 0.7000000000000001,
            "n": 36,
            "confidence": 0.6525812745094299,
            "accuracy": 0.4444444444444444
          },
          {
            "low": 0.7000000000000001,
            "high": 0.8,
            "n": 26,
            "confidence": 0.7448886632919312,
            "accuracy": 0.6153846153846154
          },
          {
            "low": 0.8,
            "high": 0.9,
            "n": 20,
            "confidence": 0.8472137451171875,
            "accuracy": 0.5
          },
          {
            "low": 0.9,
            "high": 1.0,
            "n": 7,
            "confidence": 0.9387833476066589,
            "accuracy": 0.8571428571428571
          }
        ],
        "ece_available": 0.17331071581159319,
        "selected_threshold": 0.6,
        "selected_coverage": 0.5085714285714286,
        "selected_accuracy": 0.5393258426966292,
        "selected_wrong": 41,
        "class_recall": [
          0.45901639344262296,
          0.3448275862068966,
          0.6964285714285714
        ]
      },
      "clips_losing_frames": 142,
      "discarded_pose_frames": 3789
    },
    "last128": {
      "gru": {
        "n": 175,
        "correct": 71,
        "accuracy": 0.4057142857142857,
        "macro_f1": 0.4083025869293058,
        "confusion_with_unavailable": [
          [
            23,
            3,
            35,
            0
          ],
          [
            24,
            17,
            17,
            0
          ],
          [
            24,
            1,
            31,
            0
          ],
          [
            0,
            0,
            0,
            0
          ]
        ],
        "available": 175,
        "confidence_coverage": 0.37714285714285717,
        "confident_wrong": 36,
        "confident_accuracy": 0.45454545454545453,
        "per_subject": {
          "Beginner3": {
            "n": 85,
            "correct": 23,
            "accuracy": 0.27058823529411763,
            "macro_f1": 0.24738324473189696,
            "confusion_with_unavailable": [
              [
                8,
                0,
                23,
                0
              ],
              [
                22,
                2,
                4,
                0
              ],
              [
                12,
                1,
                13,
                0
              ],
              [
                0,
                0,
                0,
                0
              ]
            ],
            "available": 85,
            "confidence_coverage": 0.4,
            "confident_wrong": 28,
            "confident_accuracy": 0.17647058823529413
          },
          "Beginner4": {
            "n": 90,
            "correct": 48,
            "accuracy": 0.5333333333333333,
            "macro_f1": 0.5422084204008978,
            "confusion_with_unavailable": [
              [
                15,
                3,
                12,
                0
              ],
              [
                2,
                15,
                13,
                0
              ],
              [
                12,
                0,
                18,
                0
              ],
              [
                0,
                0,
                0,
                0
              ]
            ],
            "available": 90,
            "confidence_coverage": 0.35555555555555557,
            "confident_wrong": 8,
            "confident_accuracy": 0.75
          }
        },
        "mean_subject_accuracy": 0.4019607843137255,
        "log_loss_available": 1.233370304107666,
        "brier_available": 0.71062443042665,
        "reliability_bins": [
          {
            "low": 0.30000000000000004,
            "high": 0.4,
            "n": 9,
            "confidence": 0.37062034010887146,
            "accuracy": 0.4444444444444444
          },
          {
            "low": 0.4,
            "high": 0.5,
            "n": 48,
            "confidence": 0.46153149008750916,
            "accuracy": 0.3958333333333333
          },
          {
            "low": 0.5,
            "high": 0.6000000000000001,
            "n": 52,
            "confidence": 0.5474387407302856,
            "accuracy": 0.34615384615384615
          },
          {
            "low": 0.6000000000000001,
            "high": 0.7000000000000001,
            "n": 36,
            "confidence": 0.6451172828674316,
            "accuracy": 0.3888888888888889
          },
          {
            "low": 0.7000000000000001,
            "high": 0.8,
            "n": 19,
            "confidence": 0.7336547374725342,
            "accuracy": 0.5789473684210527
          },
          {
            "low": 0.8,
            "high": 0.9,
            "n": 7,
            "confidence": 0.8468421697616577,
            "accuracy": 0.2857142857142857
          },
          {
            "low": 0.9,
            "high": 1.0,
            "n": 4,
            "confidence": 0.9201705455780029,
            "accuracy": 0.75
          }
        ],
        "ece_available": 0.1774684716973986,
        "selected_threshold": 0.6,
        "selected_coverage": 0.37714285714285717,
        "selected_accuracy": 0.45454545454545453,
        "selected_wrong": 36,
        "class_recall": [
          0.3770491803278688,
          0.29310344827586204,
          0.5535714285714286
        ]
      },
      "hybrid": {
        "n": 175,
        "correct": 88,
        "accuracy": 0.5028571428571429,
        "macro_f1": 0.4757994473353597,
        "confusion_with_unavailable": [
          [
            42,
            0,
            19,
            0
          ],
          [
            39,
            11,
            8,
            0
          ],
          [
            21,
            0,
            35,
            0
          ],
          [
            0,
            0,
            0,
            0
          ]
        ],
        "available": 175,
        "confidence_coverage": 0.5314285714285715,
        "confident_wrong": 45,
        "confident_accuracy": 0.5161290322580645,
        "per_subject": {
          "Beginner3": {
            "n": 85,
            "correct": 40,
            "accuracy": 0.47058823529411764,
            "macro_f1": 0.3850574712643678,
            "confusion_with_unavailable": [
              [
                21,
                0,
                10,
                0
              ],
              [
                25,
                0,
                3,
                0
              ],
              [
                7,
                0,
                19,
                0
              ],
              [
                0,
                0,
                0,
                0
              ]
            ],
            "available": 85,
            "confidence_coverage": 0.6235294117647059,
            "confident_wrong": 28,
            "confident_accuracy": 0.4716981132075472
          },
          "Beginner4": {
            "n": 90,
            "correct": 48,
            "accuracy": 0.5333333333333333,
            "macro_f1": 0.5338547562690817,
            "confusion_with_unavailable": [
              [
                21,
                0,
                9,
                0
              ],
              [
                14,
                11,
                5,
                0
              ],
              [
                14,
                0,
                16,
                0
              ],
              [
                0,
                0,
                0,
                0
              ]
            ],
            "available": 90,
            "confidence_coverage": 0.4444444444444444,
            "confident_wrong": 17,
            "confident_accuracy": 0.575
          }
        },
        "mean_subject_accuracy": 0.5019607843137255,
        "log_loss_available": 1.1932986974716187,
        "brier_available": 0.6657216282281365,
        "reliability_bins": [
          {
            "low": 0.30000000000000004,
            "high": 0.4,
            "n": 3,
            "confidence": 0.377029150724411,
            "accuracy": 0.3333333333333333
          },
          {
            "low": 0.4,
            "high": 0.5,
            "n": 31,
            "confidence": 0.458706259727478,
            "accuracy": 0.41935483870967744
          },
          {
            "low": 0.5,
            "high": 0.6000000000000001,
            "n": 48,
            "confidence": 0.5437015295028687,
            "accuracy": 0.5416666666666666
          },
          {
            "low": 0.6000000000000001,
            "high": 0.7000000000000001,
            "n": 35,
            "confidence": 0.6569556593894958,
            "accuracy": 0.5714285714285714
          },
          {
            "low": 0.7000000000000001,
            "high": 0.8,
            "n": 41,
            "confidence": 0.737006425857544,
            "accuracy": 0.5121951219512195
          },
          {
            "low": 0.8,
            "high": 0.9,
            "n": 15,
            "confidence": 0.8473836183547974,
            "accuracy": 0.3333333333333333
          },
          {
            "low": 0.9,
            "high": 1.0,
            "n": 2,
            "confidence": 0.9594895839691162,
            "accuracy": 1.0
          }
        ],
        "ece_available": 0.1225779518059322,
        "selected_threshold": 0.6,
        "selected_coverage": 0.5314285714285715,
        "selected_accuracy": 0.5161290322580645,
        "selected_wrong": 45,
        "class_recall": [
          0.6885245901639344,
          0.1896551724137931,
          0.625
        ]
      },
      "clips_losing_frames": 142,
      "discarded_pose_frames": 3789
    },
    "full_timeline": {
      "gru": {
        "n": 175,
        "correct": 84,
        "accuracy": 0.48,
        "macro_f1": 0.47888412253203544,
        "confusion_with_unavailable": [
          [
            19,
            3,
            38,
            1
          ],
          [
            17,
            24,
            17,
            0
          ],
          [
            13,
            2,
            41,
            0
          ],
          [
            0,
            0,
            0,
            0
          ]
        ],
        "available": 174,
        "confidence_coverage": 0.48,
        "confident_wrong": 40,
        "confident_accuracy": 0.5238095238095238,
        "per_subject": {
          "Beginner3": {
            "n": 85,
            "correct": 32,
            "accuracy": 0.3764705882352941,
            "macro_f1": 0.3520005286285393,
            "confusion_with_unavailable": [
              [
                7,
                3,
                20,
                1
              ],
              [
                17,
                5,
                6,
                0
              ],
              [
                4,
                2,
                20,
                0
              ],
              [
                0,
                0,
                0,
                0
              ]
            ],
            "available": 84,
            "confidence_coverage": 0.4823529411764706,
            "confident_wrong": 27,
            "confident_accuracy": 0.34146341463414637
          },
          "Beginner4": {
            "n": 90,
            "correct": 52,
            "accuracy": 0.5777777777777777,
            "macro_f1": 0.5903661464585834,
            "confusion_with_unavailable": [
              [
                12,
                0,
                18,
                0
              ],
              [
                0,
                19,
                11,
                0
              ],
              [
                9,
                0,
                21,
                0
              ],
              [
                0,
                0,
                0,
                0
              ]
            ],
            "available": 90,
            "confidence_coverage": 0.4777777777777778,
            "confident_wrong": 13,
            "confident_accuracy": 0.6976744186046512
          }
        },
        "mean_subject_accuracy": 0.4771241830065359,
        "log_loss_available": 1.144784370313058,
        "brier_available": 0.6567868677201242,
        "reliability_bins": [
          {
            "low": 0.30000000000000004,
            "high": 0.4,
            "n": 3,
            "confidence": 0.36080626646677655,
            "accuracy": 0.3333333333333333
          },
          {
            "low": 0.4,
            "high": 0.5,
            "n": 35,
            "confidence": 0.45630998866898675,
            "accuracy": 0.4
          },
          {
            "low": 0.5,
            "high": 0.6000000000000001,
            "n": 52,
            "confidence": 0.5434196006793243,
            "accuracy": 0.4807692307692308
          },
          {
            "low": 0.6000000000000001,
            "high": 0.7000000000000001,
            "n": 37,
            "confidence": 0.6465485192633964,
            "accuracy": 0.4594594594594595
          },
          {
            "low": 0.7000000000000001,
            "high": 0.8,
            "n": 29,
            "confidence": 0.7499097750104707,
            "accuracy": 0.5862068965517241
          },
          {
            "low": 0.8,
            "high": 0.9,
            "n": 14,
            "confidence": 0.8463428446224758,
            "accuracy": 0.42857142857142855
          },
          {
            "low": 0.9,
            "high": 1.0,
            "n": 4,
            "confidence": 0.9244554042816162,
            "accuracy": 1.0
          }
        ],
        "ece_available": 0.13294106053894966,
        "selected_threshold": 0.6,
        "selected_coverage": 0.48,
        "selected_accuracy": 0.5238095238095238,
        "selected_wrong": 40,
        "class_recall": [
          0.3114754098360656,
          0.41379310344827586,
          0.7321428571428571
        ]
      },
      "hybrid": {
        "n": 175,
        "correct": 83,
        "accuracy": 0.4742857142857143,
        "macro_f1": 0.4563469424139776,
        "confusion_with_unavailable": [
          [
            38,
            0,
            22,
            1
          ],
          [
            37,
            12,
            9,
            0
          ],
          [
            22,
            1,
            33,
            0
          ],
          [
            0,
            0,
            0,
            0
          ]
        ],
        "available": 174,
        "confidence_coverage": 0.5771428571428572,
        "confident_wrong": 47,
        "confident_accuracy": 0.5346534653465347,
        "per_subject": {
          "Beginner3": {
            "n": 85,
            "correct": 40,
            "accuracy": 0.47058823529411764,
            "macro_f1": 0.38547638547638546,
            "confusion_with_unavailable": [
              [
                17,
                0,
                13,
                1
              ],
              [
                24,
                0,
                4,
                0
              ],
              [
                2,
                1,
                23,
                0
              ],
              [
                0,
                0,
                0,
                0
              ]
            ],
            "available": 84,
            "confidence_coverage": 0.6470588235294118,
            "confident_wrong": 27,
            "confident_accuracy": 0.509090909090909
          },
          "Beginner4": {
            "n": 90,
            "correct": 43,
            "accuracy": 0.4777777777777778,
            "macro_f1": 0.48059964726631393,
            "confusion_with_unavailable": [
              [
                21,
                0,
                9,
                0
              ],
              [
                13,
                12,
                5,
                0
              ],
              [
                20,
                0,
                10,
                0
              ],
              [
                0,
                0,
                0,
                0
              ]
            ],
            "available": 90,
            "confidence_coverage": 0.5111111111111111,
            "confident_wrong": 20,
            "confident_accuracy": 0.5652173913043478
          }
        },
        "mean_subject_accuracy": 0.4741830065359477,
        "log_loss_available": 1.169143201270686,
        "brier_available": 0.6606459979664124,
        "reliability_bins": [
          {
            "low": 0.4,
            "high": 0.5,
            "n": 27,
            "confidence": 0.46517316269239894,
            "accuracy": 0.37037037037037035
          },
          {
            "low": 0.5,
            "high": 0.6000000000000001,
            "n": 46,
            "confidence": 0.5426731755774792,
            "accuracy": 0.41304347826086957
          },
          {
            "low": 0.6000000000000001,
            "high": 0.7000000000000001,
            "n": 41,
            "confidence": 0.6471584239384023,
            "accuracy": 0.4878048780487805
          },
          {
            "low": 0.7000000000000001,
            "high": 0.8,
            "n": 39,
            "confidence": 0.7488676619071227,
            "accuracy": 0.6153846153846154
          },
          {
            "low": 0.8,
            "high": 0.9,
            "n": 17,
            "confidence": 0.85021431481137,
            "accuracy": 0.47058823529411764
          },
          {
            "low": 0.9,
            "high": 1.0,
            "n": 4,
            "confidence": 0.9463274478912354,
            "accuracy": 0.5
          }
        ],
        "ece_available": 0.16379844142798464,
        "selected_threshold": 0.6,
        "selected_coverage": 0.5771428571428572,
        "selected_accuracy": 0.5346534653465347,
        "selected_wrong": 47,
        "class_recall": [
          0.6229508196721312,
          0.20689655172413793,
          0.5892857142857143
        ]
      },
      "clips_losing_frames": 0,
      "discarded_pose_frames": 0
    }
  },
  "coach_status": {
    "selected_beginner_clips": 24,
    "pending_coach_clips": 6,
    "prior_diagnostic_overlap": 24,
    "coach_c_annotations_pending": true
  },
  "sources": {
    "models/serve_v2_fixed/cv_results.json": {
      "sha256": "207faa1648b273ba5689e5716ed3ee8ec3beb79c17e002ece17d0a1cfc142aec"
    },
    "models/serve_v2_fixed/evaluation.json": {
      "sha256": "d7cc1e862eb678eb4c701210282423d6527bb208cf28e7083d643c32fe25d2f1"
    },
    "outputs/serve_study_v3/hybrid/results.json": {
      "sha256": "57461e6d02d3e054325863b4cd9b9c8b5e0855f79f17109991998bd64fa81ef3"
    },
    "models/serve_v2/cv_results.json": {
      "sha256": "81828e81abc03dd1889de9ba7f07781997cf2cc6ce3e6a65bbaa84f7fbfe4e90"
    },
    "models/serve_study_v3/diagnostic_evaluation.json": {
      "sha256": "0a0d1be727a49f001ae090aa1a97a6cf264d34a1ddaab70a99a18372a9cd1145",
      "download": "assets/sources/v3-diagnostic.json"
    },
    "models/serve_study_v3/angles_control/cv_results.json": {
      "sha256": "f8e9276373eb5858112a073703ee6aeb74a29a39621b4b59475bce952bd59ce1"
    },
    "models/serve_study_v3/angles_thesis_aug/cv_results.json": {
      "sha256": "03852b63f0cea52db59de72cc119a5071df329a1a4be5b9288a381716b04bcd4"
    },
    "models/serve_study_v3/skeleton/cv_results.json": {
      "sha256": "f20b66ff4594be51239d3c6828b0e518cdf21ad45aad4d7d61f03deae9266052"
    },
    "models/serve_study_v3/skeleton_motion/cv_results.json": {
      "sha256": "480d6d5f23db6dd91f5fbb0ad1bf8d75013f430cf0d0db2c34a77becd885713b"
    },
    "outputs/serve_refinement_v4/diagnostic_results.json": {
      "sha256": "03f9645559cc797eaf98c3fc326c4eef099a49a0bcef4472d41bc4fc88d127a1",
      "download": "assets/sources/v4-diagnostic.json"
    },
    "models/serve_refinement_v4/angles_thesis_aug/cv_results.json": {
      "sha256": "43c16c3f01dcde9f09eaf65081abd5788ca7548df627ae37073f51f83772fc06"
    },
    "models/serve_refinement_v4/lean_control/cv_results.json": {
      "sha256": "d241b14d1ce1a40fd1a599ed43f716df0b796920121df05b91e6e3c60075015d"
    },
    "models/serve_refinement_v4/lean_thesis_aug/cv_results.json": {
      "sha256": "d933de1eaa234f9e1a4a6703c7350c403762d45fa08db80108576f361da40a22"
    },
    "models/serve_refinement_v4/skeleton_thesis_aug/cv_results.json": {
      "sha256": "70efacf4b7b026c0f6efc6f681cc31bc0bdd1bfea7dfb684693514d0a4e95519"
    },
    "models/serve_refinement_v4/skeleton_motion_thesis_aug/cv_results.json": {
      "sha256": "f5eb46eff9b839d9849a2fc047244921089d9f1536f3b16c7c50e3f41354e2f1"
    },
    "runs/detect/ball_yolo26s/results.csv": {
      "sha256": "1dea5480870eef6ff4e825652dd61f8a201888d53b723b6582b657fa1567ab39",
      "download": "assets/sources/ball-detector.csv"
    },
    "runs/paddle_yolo26s/results.csv": {
      "sha256": "afa724c77262ff355b08934dd13bceab59edfc7b942adf84d393f12b66901d04",
      "download": "assets/sources/paddle-detector-base.csv"
    },
    "runs/detect/paddle_ft/results.csv": {
      "sha256": "582cd09e76151a56c695e690720bd1be9accb83f8cf024b8195e935bf3641c11",
      "download": "assets/sources/paddle-detector-fine-tuned.csv"
    },
    "runs/pose/court_yolo26s/results.csv": {
      "sha256": "5d6339400c888bfca48380e62df1c5bb341ec981f5c0ef1fce45a8b7af0242e2",
      "download": "assets/sources/court-landmarks-base.csv"
    },
    "runs/pose/court_ft/results.csv": {
      "sha256": "fe4817fa856ff76eba00b87a61170880b024f1a228b90fd15e96d93a068800e7",
      "download": "assets/sources/court-landmarks-fine-tuned.csv"
    },
    "outputs/serve_v2_smoke/report.json": {
      "sha256": "a50f50c7ae154358f4fb79035f612866b18885f234c98c5389546c23b6aaa618",
      "download": "assets/sources/drive-01-saved-report.json"
    },
    "outputs/research_site/samples/Beginner1_Lob_023/report.json": {
      "sha256": "f5fef73209c9b94865280d20c2ac0bdd921df8cf78f96389dec21b227085975d",
      "download": "assets/sources/lob-23-saved-report.json"
    },
    "outputs/research_site/samples/Beginner1_Topspin_022/report.json": {
      "sha256": "2d0de1d14d52e3ddcd9de9842fc05cbff2283813a2f7a28b1e3b03e2754dd9ca",
      "download": "assets/sources/topspin-22-saved-report.json"
    },
    "outputs/desktop/cf549b29380f4ca7b55b01eb0a7030f4/report.json": {
      "sha256": "77cc829e2f876608702648afba1a995c089b262e0aa5328e1e667cbbb80b9734",
      "download": "assets/sources/drive-07-saved-report.json"
    },
    "analysis/THESIS_CHAPTERS_5_6_DRAFT.md": {
      "sha256": "66f6e0d74883bd472c61c5eac1b4475fcfe6e42886400119902b06d8f9b57fca",
      "download": "assets/sources/chapters-5-6-draft.md"
    },
    "analysis/THESIS_CHAPTERS_5_6_DRAFT.docx": {
      "sha256": "7edc9d3f21c4bf88a554499ae83b601ff95a797e4481eceadaae4c3e113ddc21",
      "download": "assets/sources/chapters-5-6-draft.docx"
    },
    "docs/RESULTS_SUMMARY.md": {
      "sha256": "f12e772dfe2cc68b5c8604047b3b0555825c5f43b311ca11b133271616cce77e",
      "download": "assets/sources/original-experiments.md"
    },
    "analysis/POSE_CLASSIFIER_AUDIT.md": {
      "sha256": "a53b173168faf0c0e0326563bc0c633fd5db605f44ba3d5358ac24e9286c76c1",
      "download": "assets/sources/pose-audit.md"
    },
    "analysis/HYBRID_CLASSIFIER_AUDIT.md": {
      "sha256": "d79294fc1c2b404169afdde3345c54f305b25757f912e81d404c61fdb3bb2cab",
      "download": "assets/sources/hybrid-audit.md"
    },
    "analysis/GRU_RETRAINING_V2.md": {
      "sha256": "4605a15eabc3e98c0ab9c029a9524bcfbcc482642d0d6e0fad1af879015e13d7",
      "download": "assets/sources/gru-retraining.md"
    },
    "analysis/FOUR_FOLLOWUP_STUDY.md": {
      "sha256": "d13714b4061627b5c2593f5e8d6af9149d4c983fa6371636baa3765ee850011b",
      "download": "assets/sources/augmentation-study.md"
    },
    "analysis/SERVE_REFINEMENT_V4.md": {
      "sha256": "66864993938df190278043fd6af5fd57e7661c29712b1e71d0f3e608f55ab062",
      "download": "assets/sources/refinement-study.md"
    },
    "outputs/serve_study_v3/comparison.png": {
      "sha256": "0149fa7dd1302d8c9223ac6b3822208ab539fb524d897e841a993fd06c9d2ea3",
      "download": "assets/sources/augmentation-comparison.png"
    },
    "outputs/serve_refinement_v4/comparison.png": {
      "sha256": "5f98b1110dd00712f397a4b5aef1fec92d332d7a0ffcbc3a72a42b557059ea6c",
      "download": "assets/sources/refinement-comparison.png"
    },
    "docs/Pascua-Leones_Thesis.md": {
      "sha256": "3023552644ef272211b42ad2183f15bf0e6e7c47829dac0021b909e81bfe7622",
      "download": "assets/sources/thesis.md"
    },
    "outputs/coach_c_evaluation_v2/status.json": {
      "sha256": "e4d62af67f1323a1ad472f64d7a417913dca1fb5f4cc93f6009b768545d9ce31"
    }
  }
};

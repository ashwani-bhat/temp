### Annotation Example
[1] Key: turn | Value: {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31}
[2] Key: type | Value: {'hybrid', 'self-contagion', 'latent', 'no-context', 'inter-personal'}
[3] Key: speaker | Value: {'A', 'B'}
[4] Key: emotion | Value: {'neutral', 'happiness', 'anger', 'angry', 'excited', 'fear', 'surprise', 'disgust', 'sad', 'sadness', 'happy', 'surprised'}

### Classification Example
[...] -> this is not in the dataset, just labels in order to understand

## with context
[Example:]  "[EMOTION] anger <SEP> [Ut] No , mine ! <SEP> [Ui] Where ? my car ? <SEP> [ConvHistory] Hey , look out ! What happened ? You've just scratched my car . Oh , God , a paint was scratched off . Where ? my car ? No , mine !" [LABEL] 0, [ID] 'dailydialog_tr_10099_utt_5_impossible_cause_utt_4']

## without context
[Example:]  "[EMOTION] anger <SEP> [Ut] No , mine ! <SEP> [Ui] Where ? my car ?"[LABEL] 0,  [ID]'dailydialog_tr_10099_utt_5_impossible_cause_utt_4']


=== The QA dataset is similar to squad dataset ===
### QA example
## with context
[Context:] Conversational history (concatenation of all utterances in CH)
[Question:] The target utterance is < Ut >. The evidence utterance is < Ui >. What is the causal span from evidence in the context that is relevant to the target utterance’s emotion < Et >?
[Answer:] Causal Span

[FORMAT:] {context: "concatenation of entire conversational history",
        qas: [id: , is_impossible: , question: "same as the above question",
        answers: [text: , answer_start: ]
        }

## without context
[Context:] Causal utterance.
[Question:] The target utterance is < Ut >. What is the causal span from context that is relevant to the target utterance’s emotion < Et >?
[Answer:] Causal Span

[FORMAT:] {context: "concatenation of entire conversational history",
        qas: [id: , is_impossible: , question: "same as the above question",
        answers: [text: , answer_start: ]
        }

[id_example:] { dailydialog_tr_9686_utt_13_true_cause_utt_13_span_1 }

## Fold 1 statistics 
Test data length:  7224
Valid data length:  1185
Train data length:  27915
Iemocap test data length:  12385

## Fold 2 statistics
Test data length:  6290
Valid data length:  1147
Train data length:  25517
Iemocap test data length:  8490

## Fold 3 statistics
Test data length:  6290
Valid data length:  1147
Train data length:  25517
Iemocap test data length:  8490
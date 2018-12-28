From the segmentation lab the following parameter sets were used:

| run | # encoders/decoders| filter(s) |  learning rate | batch size | # epochs | steps per epoch | validation steps | workers |
| ----- | ---- | ------ | ---- | ------------------ | --- | --- | --- | --- |
| run1  | 1    | 20     | 0.05               | 50  | 10  | 200 | 50  | 2  |
| run2  | 1    | 20     | 0.05               | 50  | 20  | 200 | 50  | 2  |
| run3  | 1    | 20     | 0.05               | 100 | 20  | 200 | 50  | 2  |
| run4  | 1    | 20     | 0.05               | 100 | 50  | 200 | 50  | 2  |
| run5  | 2    | 20/40  | 0.05               | 100 | 50  | 200 | 50  | 2  |

And for these runs the following results have been collected:

| run  | average IoU for background | average IoU for other people | average IoU for hero | global average IoU |  
| ---- | ----               | ------------------  | ---                  | ---                 |
| run1 | 0.9828951983470107 | 0.10618387638037707 | 0.044549251198274686 | 0.37787610864188753 |
| run2 | 0.9803128523368182 | 0.17413905069974045 | 0.11007914667321698  | 0.4215103499032586  |
| run3 | 0.9853443167216672 | 0.13480468831835557 | 0.10498744888605738  | 0.4083788179753601  |
| run4 | 0.9821987577085073 | 0.1917738857966337  | 0.12318991573151504  | 0.432387519745552   |
| run5 | 0.9904808576661485 | 0.27309245560181955 | 0.1393159806112834   | 0.46762976462641714 |
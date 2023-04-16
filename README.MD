# anonymous

## Train the model

```bash
python main.py --cuda --do_train --do_valid --do_test --data_path data/DATASET-ind --geo GEO --tasks "1p.2p.3p.2i.3i.ip.pi.2u.up" -max_n 32 -ee -es -se
```

`GEO`: the reasoning method, `vec` for `GQE`, `box` for `Query2box`

`max_n`: max neighbors for getting embedding in inductive setting

`ee, es, se`: the type of queries. The front represents the anchor, and the back is the answer. `e` represents `emerge`, `s` for `seen`

## Test the model

```bash
python main.py --cuda --do_test --data_path data/DATASET-ind --geo GEO --tasks "1p.2p.3p.2i.3i.ip.pi.2u.up" --checkpoint_path PATH -max_n 32 -ee -es -se
```

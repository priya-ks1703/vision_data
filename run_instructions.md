Command to get a fully trained model as the result of our AutoML pipeline:

```bash 
python find_model.py --dataset final-test-dataset --seed 42 --output-path model.pth
```

Command to yield predictions for the final test dataset:

```bash 
python predict.py --dataset final-test-dataset --model-path model.pth --output-path final_test_preds.npy
```
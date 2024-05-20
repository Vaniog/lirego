# Lirego
<img src="https://github.com/Vaniog/metopt/blob/main/lab1/img/metopt-logo.png" width="100">

Optimization methods Lab №3

Team:

- Gurov Matvey M3233
- Tarasov Ivan M3233
- Farafonov Egor M3233


Function prediction and other optimization problems solution with linear regression

# Examples
Polynomial: \
![polynom](/_predictions/polynom.png)

Exponential: \
![exponent](/_predictions/exponent.png)

Trigonometry: \
![sin](/_predictions/sin.png)


## Code snippet
```
// x1 + 2*x2 = y
ds := training.NewSliceDataSet([][]float64{
  {0, 1, 2},
  {2, 0, 2},
  {3, 2, 7},
})

lm := ml.NewLinearModel(ml.Config{
  RowLen: 2,
  Loss:   ml.MSELoss{},
  Reg:    ml.EmptyRegularizator{},
  Bias:   false,
})

trainer := training.NewGreedyTrainer(0, 100000, 0.1)
trainer.Train(lm, ds)

fmt.Println(lm.Predict(mat.NewVecDense(2, []float64{1, 1})))
//Output: 2.999999
```

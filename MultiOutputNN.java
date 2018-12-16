
/**
 * Write a description of class Perceptron here.
 *
 * @author (your name)
 * @version (a version number or a date)
 */

import java.lang.*;
import java.util.Random;

public class MultiOutputNN
{
    // instance variables - replace the example below with your own
    private int inputSize;
    private double[][] inputToHiddenW;
    private double[][] hiddenToOutputW;
    private int print = 100;
    private Random rand = new Random();
    private static int NODES = 6;
    private double[] losses = new double[NODES];
    private double[][] hVec;

    /**
     * Constructor for objects of class Perceptron
     */
    public MultiOutputNN(int inputSize)
    {
        this.inputSize = inputSize;

        inputToHiddenW = new double[inputSize*2][inputSize];
        hiddenToOutputW = new double[NODES][inputSize*2];
        initializeWeights();
        System.out.println("Weights initialized to: ");
        printWeights();
    }
    // this method does feed forward and backprop for all training inputs for a
    // given number of epochs
    public void trainNN(double[][] inputs, int[] target, double lr, int epochs){
        double[][] hVec;
        int counter = 0;

        double totalLoss= 0.0;

        for(int epoch = 0; epoch < epochs; epoch++){
          int input_length = inputs.length;

          //printWeights();
          double loss = 0.0;

          for(int i = 0; i < input_length; i++){
              hVec = feedForward(inputs[i]);
              loss = backpropagate(hVec, target[i], lr, inputs[i]);
              totalLoss += loss;
              if(counter % print == 0){//note
                  double avg = totalLoss / print;
                  System.out.println("Here's the loss for epoch: " + epoch + " at iteration" + counter + " : " + avg);
                  totalLoss = 0;
                  //printWeights();
              }
              counter++;
          }

        }

    }

    public double[][] feedForward(double[] input){
      double accum = 0.0;
      double[][] hVec = new double[2][input.length*2];
        for(int i = 0; i < inputToHiddenW.length; i++){
          for(int j = 0; j < inputToHiddenW[0].length; j++){
            accum += input[j] * inputToHiddenW[i][j];
          }
          hVec[0][i] = sigmoid(accum);
        }

        accum = 0;
        for(int i = 0; i < hiddenToOutputW.length; i++){
          for(int j = 0; j < hiddenToOutputW[0].length; j++){
            accum += hVec[0][j]*hiddenToOutputW[i][j];
          }
          hVec[1][i] = sigmoid(accum);
          System.out.println("For class " + (i + 1) + " the system gives a score of: " + sigmoid(accum));
        }
        return hVec;
    }

    public double backpropagate(double[][] hVec, int target, double lr,double[] input){
        //Represents the summation required to calculate weights connecting the input and hidden layers
        double[] inputHidSum = new double[hiddenToOutputW[0].length];
        for(int i = 0; i < hiddenToOutputW.length; i++){
          if (i + 1 == target) {
              losses[i] = Math.pow(1 - hVec[1][i],2) * 0.5;
          } else {
              losses[i] = Math.pow(0 - hVec[1][i], 2) * 0.5;
          }
          double derivSigmoid = hVec[1][i] * (hVec[1][i] - 1);
          for(int j = 0; j < hiddenToOutputW[0].length; j++){
              hiddenToOutputW[i][j] = hiddenToOutputW[i][j] + (lr * hVec[0][j] * losses[i] * derivSigmoid);
              inputHidSum[j] += (hiddenToOutputW[i][j] * losses[i] * derivSigmoid);
          }
        }
        for(int i = 0; i < inputToHiddenW.length; i++){
            double derivSigmoid = hVec[0][i] * (hVec[0][i] - 1);
            for(int j = 0; j < inputToHiddenW[0].length; j++){
            inputToHiddenW[i][j] = inputToHiddenW[i][j] + (lr * input[j] * inputHidSum[i] * derivSigmoid);
          }
        }
        double totalLoss = 0.0;
        for(double loss : losses){
          totalLoss += loss;
        }
        return totalLoss;
      }

    public void calcAverages(int num, int iter){
      System.out.println("Here are the averages for iteration " + iter + ": ");
      double average;
      for(int i = 0; i < NODES; i++){
        average = losses[i]/num;
        System.out.println(i + ": " + average);
        losses[i] = 0.0;
      }

    }

    public double sigmoid(double x){
      double rawClass = 0.0;
      rawClass = 1.0/(1+Math.exp((-1 * x)));
      return rawClass;
    }

    public void initializeWeights(){
        double weight;
        // initializing the weights for the input to hidden layer
        for(int i = 0; i < inputToHiddenW.length; i++){
            for(int j = 0; j < inputToHiddenW[0].length; j++){
              weight = rand.nextDouble();
              if(weight > 0.5){
                weight = rand.nextDouble() * 0.15;
                //System.out.println(" Initialize!: " + weight);
                inputToHiddenW[i][j] = weight;
              }
              else{
                weight = rand.nextDouble() * -0.15;
                //System.out.println(" Initialize!: " + weight);
                inputToHiddenW[i][j] = weight;
              }
            }
        }
        // this initializes the weights from the hiddne layer to the output layer
        for(int i = 0; i < hiddenToOutputW.length; i++){
            for(int j = 0; j < hiddenToOutputW[0].length; j++){
              weight = rand.nextDouble();
              if(weight > 0.5){
                weight = rand.nextDouble() * 0.15;
                //System.out.println(" Initialize!: " + weight);
                hiddenToOutputW[i][j] = weight;
              }
              else{
                weight = rand.nextDouble() * -0.15;
                //System.out.println(" Initialize!: " + weight);
                hiddenToOutputW[i][j] = weight;
              }
            }
        }
    }
    public void printWeights(){
      for(int i = 0; i < inputToHiddenW.length; i++){
        System.out.println("these are the weights from in to hidden: ");
        for(int j = 0; j < inputToHiddenW[0].length; j++){
          System.out.println(j + ":\t" + inputToHiddenW[i][j]);
        }
      }

      for(int i = 0; i < hiddenToOutputW.length; i++){
        System.out.println("these are the weights from hidden to output: ");
        for(int j = 0; j < hiddenToOutputW[0].length; j++){
          System.out.println(j + ":\t" + hiddenToOutputW[i][j]);
        }
      }


    }
    /*
    public void evaluatePerceptron(double[][] test, int[] target){
      double max = 0.0;
      double classification;
      int theClass = -1;
      int numCorrect = 0;
      int numIncorrect = 0;
      for(int i = 0; i < test.length; i++){
        for(int node = 0; node < NODES; node++){
          classification = getRawClassification(node, test[i]);
          if(classification > max){
            max = classification;
            theClass = node;
          }
        }

        if(theClass == target[i]){
          numCorrect++;
        }
        else{
          numIncorrect++;
        }

      }

      System.out.println("Here's how many it got right: " + numCorrect);
      System.out.println("Here's how many it got wrong: " + numIncorrect);
    }
    */
}

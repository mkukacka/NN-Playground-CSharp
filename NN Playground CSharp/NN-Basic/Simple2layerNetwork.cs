using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN_Basic
{
    public class Simple2layerNetwork
    {

        private int inputSize;
        private int hiddenLayerSize;
        private int outputSize;

        private double activationParameter;

        private Random random;
        private double [][][] w;            // weights, indexes: layer, neuron, neuron in lower layer/input
        private double [][] biases;         // biases of neurons, indexes: layer, neuron
        private double [][] activations;    // activation values of neuron, indexes: layer, neuron. Layer 0 is input, 1 is hidden layer, 2 is output layer.
        
        private double [][] errors;         // error coefficients of neurons, indexes: layer, neuron. Layer 0 are errors of hidden layer, layer 1 are errors of output layer.

        public Simple2layerNetwork(int inputSize, int hiddenLayerSize, int outputSize,
            double activationParameter, int? randomizerSeed = null)
        {
            this.inputSize = inputSize;
            this.hiddenLayerSize = hiddenLayerSize;
            this.outputSize = outputSize;
            this.activationParameter = activationParameter;

            if (randomizerSeed.HasValue)
            {
                this.random = new Random(randomizerSeed.Value);
            }
            else
            {
                this.random = new Random();
            }

            InitWeights();
        }

        private void InitWeights()
        {
            w = new double[2][][];
            biases = new double[2][];
            errors = new double[2][];

            w[0] = new double[hiddenLayerSize][];
            biases[0] = new double[hiddenLayerSize];
            errors[0] = new double[hiddenLayerSize];

            for (int i=0; i<hiddenLayerSize; i++)
            {
                w[0][i] = new double[inputSize];
                for (int j = 0; j < inputSize; j++)
                {
                    w[0][i][j] = GetRandomWeight();
                }

                biases[0][i] = GetRandomWeight();
            }
            

            w[1] = new double[outputSize][];
            biases[1] = new double[outputSize];
            
            errors[1] = new double[outputSize];
            for (int i = 0; i < outputSize; i++)
            {
                w[1][i] = new double[hiddenLayerSize];
                for (int j = 0; j < hiddenLayerSize; j++)
                {
                    w[1][i][j] = GetRandomWeight();
                }

                biases[1][i] = GetRandomWeight();
            }

            activations = new double[3][];
            activations[0] = new double[inputSize];
            activations[1] = new double[hiddenLayerSize];
            activations[2] = new double[outputSize];

        }

        public double[] GetNetworkOutput(double [] input)
        {
            VerifyInput(input);

            // comput network activations
            ComputeNetworkActivations(input);

            // return the second layer activations
            var res = new double[this.outputSize];
            Array.Copy(activations[2], res, this.outputSize);
            return res;
        }

        

        private void ComputeNetworkActivations(double [] input)
        {
            SetInputValues(input);

            // traverse all layers from bottom, all neurons in the layer, all connections of that neuron 
            // - add weighted inputs (and bias) and run resulting potential through activation function
            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < w[i].Length; j++)
                {
                    double neuronPotential = 0.0;

                    for (int k = 0; k < w[i][j].Length; k++)
                    {
                        neuronPotential += w[i][j][k] * activations[i][k];
                    }
                    neuronPotential += biases[i][j];    // do not forget the bias

                    activations[i+1][j] = ActivationFunction(neuronPotential);
                }
            }
        }

        private void SetInputValues(double[] input)
        {
            for(int i=0; i < this.inputSize; ++i)
            {
                activations[0][i] = input[i];
            }
        }

        public double GetNetworkTotalError(double [] input, double [] targetOutput)
        {
            VerifyInput(input);
            VerifyTargetOutput(targetOutput);

            // compute network activations
            ComputeNetworkActivations(input);

            // measure the error between output and target output
            return GetCurrentNetworkError(targetOutput);
        }

        private double GetCurrentNetworkError(double[] targetOutput)
        {
            // compute difference between current output activations and target output
            double err = 0.0;

            double tmp;
            for (int i = 0; i < outputSize; ++i)
            {
                tmp = activations[2][i] - targetOutput[i];
                err += tmp * tmp;
            }
            err /= 2.0;

            return err;
        }

        public void Learn(double [] input, double [] targetOutput, double learningParam)
        {
            VerifyInput(input);
            VerifyTargetOutput(targetOutput);

            // compute activations
            ComputeNetworkActivations(input);

            // measure error on output layer - per neuron - and assign it to the neuron's error coef
            for (int i = 0; i < outputSize; i++)
            {
                errors[1][i] = (activations[2][i] - targetOutput[i])                                    // derivation of the error function
                               * activationParameter * activations[2][i] * (1.0 - activations[2][i])    // derivation of the activation function
                    ;
            }

            // propagate the error via the weights to the hidden layer
            for (int i = 0; i < hiddenLayerSize; i++)
            {
                errors[0][i] = 0.0;
                for (int j = 0; j < outputSize; j++)
                {
                    errors[0][i] += errors[1][j] * w[1][j][i];
                }
                errors[0][i] *= activationParameter * activations[1][i] * (1.0 - activations[1][i]);    // derivation of the activation function
            }

            // adapt weights
            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < w[i].Length; j++)
                {
                    for (int k = 0; k < w[i][j].Length; k++)
                    {
                        w[i][j][k] += -1.0 * learningParam      // reversing derivation and adding factor managing the step size
                                      * errors[i][j]            // times the error on the neuron
                                      * activations[i][k];      // times the activation of the neuron in the lower layer/input
                    }

                    biases[i][j] += -1.0 * learningParam * errors[i][j];    // bias works as a weight to a neuron with output of constant 1.0
                }
            }


        }

        #region Verification methods
        private void VerifyInput(double[] input)
        {
            if (input.Length != this.inputSize)
            {
                throw new Exception(string.Format("Input dimension {0} does not match network input layer dimension {1}", input.Length, this.inputSize));
            }
        }

        private void VerifyTargetOutput(double[] targetOutput)
        {
            if (targetOutput.Length != this.outputSize)
            {
                throw new Exception(string.Format("Target output dimension {0} does not match network output layer dimension {1}", targetOutput.Length, this.outputSize));
            }
        }
        #endregion

        private double ActivationFunction(double potential)
        {
            return 1.0 / (1.0 + Math.Exp(-1.0 * activationParameter * potential));
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns>Random value between -1.0 and 1.0</returns>
        private double GetRandomWeight()
        {
            return random.NextDouble() * 2.0 - 1.0;
        }

    }
}

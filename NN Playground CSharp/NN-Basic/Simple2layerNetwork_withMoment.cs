using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN_Basic
{
    /// <summary>
    /// Simple 2-layer feed-forward neural network, with added learning moment, for more rapid learning convergence.
    /// </summary>
    public class Simple2layerNetwork_withMoment : Simple2layerNetwork
    {
        protected double[][][] pw;            // weights from the previous learning step, indexes: layer, neuron, neuron in lower layer/input
        protected double[][] pb;         // biases from previous learning step, indexes: layer, neuron
        protected double learningMoment;

        public Simple2layerNetwork_withMoment(int inputSize, int hiddenLayerSize, int outputSize, double activationParameter, int? randomizerSeed = null,
            double learningMoment = 0.0) 
            : base(inputSize, hiddenLayerSize, outputSize, activationParameter, randomizerSeed)
        {
            this.learningMoment = learningMoment;

            // after weights have been initalized by the superclass ctor, copy them to the pw array
            pw = new double[w.Length][][];
            for (int i = 0; i < w.Length; i++)
            {
                pw[i] = new double[w[i].Length][];
                for (int j = 0; j < w[i].Length; j++)
                {
                    pw[i][j] = new double[w[i][j].Length];
                    for (int k = 0; k < w[i][j].Length; k++)
                    {
                        pw[i][j][k] = w[i][j][k];
                    }
                }
            }

            // also biases
            pb = new double[biases.Length][];
            for (int i = 0; i < biases.Length; i++)
            {
                pb[i] =new double[biases[i].Length];
                for (int j = 0; j < biases[i].Length; j++)
                {
                    pb[i][j] = biases[i][j];
                }
            }
        }

        protected override void AdaptWeightsBasedOnErrors(double learningParam)
        {
            // adapt weights
            double tmpWeight;
            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < w[i].Length; j++)
                {

                    for (int k = 0; k < w[i][j].Length; k++)
                    {
                        tmpWeight = w[i][j][k];
                        w[i][j][k] += -1.0 * learningParam      // reversing derivation and adding factor managing the step size
                                      * errors[i][j]            // times the error on the neuron
                                      * activations[i][k]       // times the activation of the neuron in the lower layer/input
                                      + learningMoment * (w[i][j][k] - pw[i][j][k])
                            ;

                        pw[i][j][k] = tmpWeight;
                    }

                    tmpWeight = biases[i][j];
                    biases[i][j] += -1.0 * learningParam * errors[i][j] // bias works as a weight to a neuron with output of constant 1.0
                                    + learningMoment * (biases[i][j] - pb[i][j])
                        ;
                    pb[i][j] = tmpWeight;
                }
            }
        }


    }
}

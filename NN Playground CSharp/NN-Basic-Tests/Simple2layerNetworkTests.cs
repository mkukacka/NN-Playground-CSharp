using NN_Basic;
using NUnit.Framework;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN_Basic_Tests
{
    [TestFixture]
    public class Simple2layerNetworkTests
    {

        [Test]
        public void SimpleNetworkProcessingTest()
        {
            // just to see whether everything is going ok - basic implementation check
            
            var trainingSet = new List<Tuple<double[], double[]>>{
                new Tuple<double[], double[]>(new double[]{0.0, 0.5, 1.0 }, new double[]{1.0, 0.0, 0.0}),
                new Tuple<double[], double[]>(new double[]{1.0, 0.5, 0.0 }, new double[]{0.0, 1.0, 0.0}),
                new Tuple<double[], double[]>(new double[]{0.0, 0.1, 0.0 }, new double[]{0.0, 0.0, 1.0})
            };

            var inputSize = 3;
            var outputSize = 3;
            var hiddenLayerSize = 4;
            var activationParameter = 0.5;

            var network = new Simple2layerNetwork(inputSize, hiddenLayerSize, outputSize, activationParameter);

            var output = network.GetNetworkOutput(trainingSet[0].Item1);
        }

        [Test]
        public void SimpleLearningTest()
        {
            var trainingSet = new List<Tuple<double[], double[]>>{
                new Tuple<double[], double[]>(new double[]{0.0, 0.5, 1.0 }, new double[]{1.0, 0.0, 0.0}),
                new Tuple<double[], double[]>(new double[]{1.0, 0.5, 0.0 }, new double[]{0.0, 1.0, 0.0}),
                new Tuple<double[], double[]>(new double[]{0.0, 0.1, 0.0 }, new double[]{0.0, 0.0, 1.0})
            };

            var inputSize = 3;
            var outputSize = 3;
            var hiddenLayerSize = 4;
            var activationParameter = 1.0;

            var learningParam = 1.0;

            var network = new Simple2layerNetwork(inputSize, hiddenLayerSize, outputSize, activationParameter);

            int numLearningCycles = 50;

            double err;
            for (int i = 0; i < numLearningCycles; i++)
            {
                // measure the error before the learning, on the whole training set
                err = 0.0;
                foreach (var trainSample in trainingSet)
                {
                    err += network.GetNetworkTotalError(trainSample.Item1, trainSample.Item2);
                }
                Console.WriteLine("Step {0}, total network error: {1}", i, err);

                // train all samples
                foreach (var trainSample in trainingSet)
                {
                    network.Learn(trainSample.Item1, trainSample.Item2, learningParam);
                }
            }

            // measure final error of the network
            err = 0.0;
            foreach (var trainSample in trainingSet)
            {
                err += network.GetNetworkTotalError(trainSample.Item1, trainSample.Item2);
            }
            Console.WriteLine("Final total network error: {0}", err);
        }
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN_Common.Interfaces
{
    public interface IFeedForwardNeuralNetwork
    {
        int InputSize { get; }
        int OutputSize { get; }

        double[] GetNetworkOutput(double[] input);
    }
}

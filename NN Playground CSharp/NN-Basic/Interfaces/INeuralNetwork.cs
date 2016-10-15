using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN_Basic.Interfaces
{
    public interface IFeedForwardNeuralNetwork
    {
        int InputSize { get; }
        int OutputSize { get; }

    }
}

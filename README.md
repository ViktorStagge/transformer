# Attention is all you need
A keras implementation of the _Transformer_ in "Attention is all you need".

Written after implementing the [Compressive Transformer](http://github.com/ViktorStagge/CompressiveTransformer) (originally created by Rae et. al) - as everything was already in place.
Furthermore, the original _Transformer_ is much better suited for being implemented in keras, as compared to the Compressive Transformer. This, as it has no internal states that resides outside of the own model's computational graph, nor does it use multiple different losses for its respective models. 

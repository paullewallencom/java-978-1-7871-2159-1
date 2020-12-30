package com.packt.deeplearning.section1;

import org.apache.log4j.BasicConfigurator;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;

import java.io.IOException;
import java.text.DecimalFormat;
import java.util.Collections;

public class RecurrentNeuralNetwork {
    private final static Logger LOGGER = org.slf4j.LoggerFactory.getLogger(RecurrentNeuralNetwork.class);

    public static void main(String[] args) throws IOException, InterruptedException {
        BasicConfigurator.configure();
        DecimalFormat f = new DecimalFormat("#0.000");
        int miniBatchSize = 1;
        int labelIndex = 6;
        int numPossibleLabels = -1;
        DataNormalization normalizer = new NormalizerMinMaxScaler(0, 1);
        FileSplit trainFileSplit = new FileSplit(new ClassPathResource("data/multilinearregression/train.csv").getFile());
        FileSplit testFileSplit = new FileSplit(new ClassPathResource("data/multilinearregression/test.csv").getFile());

        CSVSequenceRecordReader trainReader = new CSVSequenceRecordReader(1, ",");
        CSVSequenceRecordReader testReader = new CSVSequenceRecordReader(1, ",");

        trainReader.initialize(trainFileSplit);
        testReader.initialize(testFileSplit);

        SequenceRecordReaderDataSetIterator trainIter = new SequenceRecordReaderDataSetIterator(trainReader, miniBatchSize, numPossibleLabels, labelIndex, true);
        SequenceRecordReaderDataSetIterator testIter = new SequenceRecordReaderDataSetIterator(testReader, miniBatchSize, numPossibleLabels, labelIndex, true);

        normalizer.fit(trainIter);
        trainIter.reset();
        trainIter.setPreProcessor(normalizer);
        testIter.setPreProcessor(normalizer);

        int seed = 1990;
        double learningRate = 1e-5;
        int nOut = 1;
        int layerSize = 10;
        int epoch = 40;
        int iterations = 1;
        int tBPTTForwardLength = 10;
        int tBPTTBackwardLength = 10;

        MultiLayerNetwork net = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
                                                              .seed(seed)
                                                              .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                                                              .iterations(iterations)
                                                              .weightInit(WeightInit.XAVIER)
                                                              .miniBatch(false)
                                                              .updater(Updater.NESTEROVS)
                                                              .momentum(0.95)
                                                              .learningRate(learningRate)
                                                              .list()
                                                              .layer(0, new GravesLSTM.Builder().activation(Activation.SOFTSIGN).nIn(trainIter.inputColumns()).nOut(layerSize)
                                                                                                .build())
                                                              .layer(1, new GravesLSTM.Builder().activation(Activation.SOFTSIGN).nIn(layerSize).nOut(layerSize)
                                                                                                .build())
                                                              .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE)
                                                                      .activation(Activation.IDENTITY).nIn(layerSize).nOut(nOut).build())
                                                              .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tBPTTForwardLength).tBPTTBackwardLength(tBPTTBackwardLength)
                                                                                              .build());
        net.init();
        net.setListeners(new ScoreIterationListener(1000));

        //Train & Evaluate the Regression Model
        for (int i = 0; i < epoch; i++) {
            trainIter.reset();
            testIter.reset();
            net.fit(trainIter);
            LOGGER.info("Epoch " + i + " complete. Time series evaluation:");
            RegressionEvaluation eval = new RegressionEvaluation(Collections.singletonList("signals"), 4);
            while (testIter.hasNext()) {
                DataSet t = testIter.next();
                INDArray features = t.getFeatures();
                INDArray labels = t.getLabels();
                INDArray predicted = net.output(features, false);
                eval.evalTimeSeries(labels, predicted);
            }
            LOGGER.info("RMSE: " + f.format(eval.rootMeanSquaredError(0)));
            LOGGER.info(eval.stats());
        }
    }
}
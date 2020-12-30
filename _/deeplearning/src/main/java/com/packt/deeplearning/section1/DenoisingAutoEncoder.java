package com.packt.deeplearning.section1;

import org.apache.log4j.BasicConfigurator;
import org.apache.log4j.xml.DOMConfigurator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.AutoEncoder;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;

public class DenoisingAutoEncoder {
    private final static Logger LOGGER = org.slf4j.LoggerFactory.getLogger(DenoisingAutoEncoder.class);

    public static void main(String[] args) throws IOException, InterruptedException {
        DOMConfigurator.configure("src/main/resources/log4j.xml");
        int labelIndex = 4;
        int numClasses = 3;
        int batchSize = 1000;
        int seed = 30;
        int iterations = 1;
        int epoch = 1000;
        int numLinesToSkip = 0;
        String delimiter = ",";
        RecordReader allDataCsv = new CSVRecordReader(numLinesToSkip, delimiter);
        FileSplit irisCSV = new FileSplit(new ClassPathResource("data/multilayerperceptron/IRIS_.csv").getFile());
        allDataCsv.initialize(irisCSV);
        DataSetIterator allDataIterator = new RecordReaderDataSetIterator(allDataCsv, batchSize, labelIndex, numClasses);

        org.nd4j.linalg.dataset.DataSet allData = allDataIterator.next();
        allData.normalize();
        allData.shuffle(seed);
        SplitTestAndTrain splitTestAndTrain = allData.splitTestAndTrain(0.80);

        org.nd4j.linalg.dataset.DataSet train = splitTestAndTrain.getTrain();
        org.nd4j.linalg.dataset.DataSet test = splitTestAndTrain.getTest();

        AutoEncoder encoder = new AutoEncoder.Builder().nIn(4).nOut(2).weightInit(WeightInit.XAVIER).lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY).corruptionLevel(0.01).build();
        AutoEncoder decoder = new AutoEncoder.Builder().nIn(2).nOut(4).weightInit(WeightInit.XAVIER).lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY).corruptionLevel(0.01).build();
        OutputLayer out = new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).activation(Activation.SOFTMAX).nIn(4).nOut(3).build();

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .momentumAfter(Collections.singletonMap(3, 0.9))
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .regularization(true)
                .l1(1e-5)
                .l2(1e-5)
                .list()
                .layer(0, encoder)
                .layer(1, decoder)
                .layer(2, out)
                .backprop(true)
                .pretrain(true)
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(1000));

        trainAndEvaluate(epoch,train,test,net);

        double[] latentVariables = net.getLayer(0).preOutput(test.getFeatures()).data().asDouble();
        int[] labels = test.getLabels().data().asInt();

        for (int i = 0; i < latentVariables.length-1; i=i+2) {
            String label = "";
            if(labels[i/2]==1){
                label = "0";
            }else if(labels[1+i/2]==1){
                label =  "1";
            }else{
                label = "2";
            }
            System.out.println(latentVariables[i] + "," + latentVariables[i + 1] + "," + label);
        }

    }

    private static void trainAndEvaluate(int epoch, DataSet train, DataSet test, MultiLayerNetwork net) {
        for (int i = 0; i < epoch; i++) {
            net.fit(train);
            LOGGER.info("Epoch " + i + " complete. Evaluation:");
            Evaluation eval = new Evaluation(Arrays.asList("Iris-versicolor", "Iris-virginica", "Iris-setosa"), 3);
            INDArray features = test.getFeatures();
            INDArray labels = test.getLabels();
            INDArray predicted = net.output(features, false);
            eval.eval(labels, predicted);
            LOGGER.info(eval.stats());
        }
    }
}

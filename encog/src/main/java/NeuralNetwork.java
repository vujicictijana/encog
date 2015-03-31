import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.text.DecimalFormat;

import org.encog.Encog;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataPair;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;

public class NeuralNetwork {
	BasicNetwork network = new BasicNetwork();

	// network
	int inputParametersNo;
	int outputParametersNo;
	int hiddenNeurons;

	// input data for training and testing
	public double input[][];
	public double inputTest[][];

	// output data for training and testing
	public double output[][];
	public double outputTest[][];

	// files for training and testing
	String testFile;
	String trainFile;

	// no of rows in files for training and testing
	int trainDataNo;
	int testDataNo;

	// training data set
	MLDataSet trainingSet;

	// test data set
	MLDataSet testSet;

	public NeuralNetwork(int inputParametersNo, int outputParametersNo,
			int hiddenNeurons, String testFile, int testDataNo,
			String trainFile, int trainDataNo) {
		super();
		this.inputParametersNo = inputParametersNo;
		this.outputParametersNo = outputParametersNo;
		this.hiddenNeurons = hiddenNeurons;
		this.testFile = testFile;
		this.trainFile = trainFile;
		input = new double[trainDataNo][inputParametersNo];
		inputTest = new double[testDataNo][inputParametersNo];
		output = new double[trainDataNo][outputParametersNo];
		outputTest = new double[testDataNo][outputParametersNo];
	}

	public void create() {

		readInput(trainFile, input);
		readOutput(trainFile, output);

		trainingSet = new BasicMLDataSet(input, output);

		// create a neural network, without using a factory

		network.addLayer(new BasicLayer(null, true, inputParametersNo));
		network.addLayer(new BasicLayer(new ActivationSigmoid(), true,
				hiddenNeurons));
		network.addLayer(new BasicLayer(new ActivationSigmoid(), false,
				outputParametersNo));
		network.getStructure().finalizeStructure();
		network.reset();

		// train the neural network
		final ResilientPropagation train = new ResilientPropagation(network,
				trainingSet);

		int epoch = 1;

		do {
			train.iteration();
			System.out
					.println("Epoch #" + epoch + " Error:" + train.getError());
			epoch++;
		} while (train.getError() > 0.01);
		System.out.println(epoch);
		train.finishTraining();
	}

	public void test() {
		readInput(testFile, inputTest);
		readOutput(testFile, outputTest);

		testSet = new BasicMLDataSet(inputTest, outputTest);

		// test the neural network
		System.out.println("Neural Network Results:");

		DecimalFormat df = new DecimalFormat("#.##");
		for (MLDataPair pair : testSet) {
			final MLData output = network.compute(pair.getInput());
			System.out
					.println(pair.getInput().getData(0)
							+ ","
							+ pair.getInput().getData(1)
							+ ", actual="
							+ df.format(output.getData(0))
							+ ",ideal="
							+ df.format(pair.getIdeal().getData(0))
							+ ", error="
							+ df.format((pair.getIdeal().getData(0) - output
									.getData(0))));
		}

		Encog.getInstance().shutdown();
	}

	public String getResult(double[] data) {
		MLDataPair pair = new BasicMLDataPair(new BasicMLData(data));

		testSet = new BasicMLDataSet(inputTest, outputTest);

		MLData output = network.compute(pair.getInput());
		double result = output.getData(0);

		Encog.getInstance().shutdown();
		if (result > 0.75) {
			return "relevant";

		} else {
			return "irrelevant";
		}
	}

	public void readInput(String fileName, double[][] data) {

		File f = new File(fileName);

		try {
			BufferedReader read = new BufferedReader(new FileReader(f));
			String s = read.readLine();
			int a = 0;
			while (s != null) {
				String[] array = s.split(",");
				for (int i = 0; i < array.length - 1; i++) {
					data[a][i] = Double.parseDouble(array[i]);
				}
				a++;
				s = read.readLine();
			}
			read.close();
		} catch (Exception e) {
			System.out.println(e.getMessage());
		}
	}

	public void readOutput(String fileName, double[][] data) {

		File f = new File(fileName);

		try {
			BufferedReader read = new BufferedReader(new FileReader(f));
			String s = read.readLine();
			int a = 0;
			while (s != null) {
				String[] array = s.split(",");
				data[a][0] = Double.parseDouble(array[array.length - 1]);
				a++;
				s = read.readLine();
			}
			read.close();
		} catch (Exception e) {
			System.out.println(e.getMessage());
		}
	}

	public void showData(double[][] data) {
		int col = data[0].length;
		for (int i = 0; i < data.length; i++) {
			System.out.print((i + 1) + " ");
			for (int j = 0; j < col; j++) {
				System.out.print(data[i][j] + "/");
			}
			System.out.println();
		}
	}
}

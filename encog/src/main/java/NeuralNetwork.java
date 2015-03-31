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
import org.encog.neural.networks.training.propagation.back.Backpropagation;

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

		// create a neural network

		network.addLayer(new BasicLayer(null, true, inputParametersNo));
		network.addLayer(new BasicLayer(new ActivationSigmoid(), true,
				hiddenNeurons));
		network.addLayer(new BasicLayer(new ActivationSigmoid(), false,
				outputParametersNo));
		network.getStructure().finalizeStructure();
		network.reset();

		// train the neural network
		// final ResilientPropagation train = new ResilientPropagation(network,
		// trainingSet);
		final Backpropagation train = new Backpropagation(network, trainingSet);

		int iteration = 1;

		do {
			train.iteration();
			System.out.println("Iteration #" + iteration + " Error:"
					+ train.getError());
			iteration++;
		} while (train.getError() > 0.01);
		System.out.println("Total iterations: " + iteration);
		train.finishTraining();
	}

	public void test() {
		readInput(testFile, inputTest);
		readOutput(testFile, outputTest);

		testSet = new BasicMLDataSet(inputTest, outputTest);

		// test the neural network
		System.out.println("Neural Network Results:");

		DecimalFormat df = new DecimalFormat("#.###");

		StringBuffer s;
		for (MLDataPair pair : testSet) {
			s = new StringBuffer();
			final MLData output = network.compute(pair.getInput());
			s.append("Input: ");
			for (int i = 0; i < pair.getInput().size(); i++) {
				s.append(pair.getInput().getData(i) + "; ");
			}
			s.append(" Output: " + df.format(output.getData(0)));
			s.append(" Desired output: "
					+ df.format(pair.getIdeal().getData(0)));
			s.append(" Error: "
					+ df.format((pair.getIdeal().getData(0) - output.getData(0))));

			System.out.println(s.toString());

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

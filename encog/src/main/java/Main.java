public class Main {

	public static void main(String[] args) {
		
		NeuralNetwork n = new NeuralNetwork(6, 1, 13, "test.csv", 10,
				"trening.csv", 40);
		System.out.println("Treniranje......");
		n.create();
		System.out.println("\nTestiranje......");
		n.test();
		System.out.println("\nProvjera za konkretnu igru 142670:");
		double[] input = {0,1,4,1,1,366};
		System.out.println(n.getResult(input));
		
		long startTime = System.nanoTime();
		n.getResult(input);
		long endTime = System.nanoTime();
		long duration = (endTime - startTime);
		System.out.println("\nTrajanje:" + (duration/1000.00) + " microseconds");
	}

}

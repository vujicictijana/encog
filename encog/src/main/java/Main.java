public class Main {

	public static void main(String[] args) {
		NeuralNetwork n = new NeuralNetwork(6, 1, 13, "test.csv", 10,
				"trening.csv", 40);
		System.out.println("Treniranje......");
		n.create();
		System.out.println("Testiranje......");
		n.test();
		System.out.println("kraj");
	}

}

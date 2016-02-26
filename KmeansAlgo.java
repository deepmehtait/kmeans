package cs286;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

/**
 * Created by Deep on 12/4/2015.
 */
public class KmeansAlgo {

	private static int k;
	private static int maxKmeanIteration;
	private static double delta;
	private static String distance;
	private static String initMethod;
	private static ClusterModel[] clusters;

	public static void main(String[] args) {

		try {
			k = Integer.parseInt(args[0]);
		} catch (NumberFormatException nfe) {
			System.err.println("Number of clusters must be an integer.");
			System.exit(1);
		}

		try {
			maxKmeanIteration = Integer.parseInt(args[1]);
		} catch (NumberFormatException nfe) {
			System.err.println("Number of iterations must be an integer.");
			System.exit(1);
		}

		try {
			delta = Double.parseDouble(args[2]);
		} catch (NumberFormatException nfe) {
			System.err.println("Delta must be a numeric value.");
			System.exit(1);
		}

		if (delta <= 0.0) {
			System.err.println("Delta must be a positive numeric value.");
			System.exit(1);
		}

		distance = args[3];
		initMethod = args[4];

		if (!("euclidean".equals(distance) || "cosine".equals(distance))) {
			System.err.println("Please specify distance as either euclidean or cosine.");
			System.exit(1);
		}

		if (!("random".equals(initMethod) || "partition".equals(initMethod))) {
			System.err.println("Please specify initialization method as either random or partition.");
			System.exit(1);
		}

		Path dataFilePath = Paths.get(args[5]);
		if (Files.notExists(dataFilePath)) {
			System.err.println("Input file does not exist on given path: " + dataFilePath.toString());
			System.exit(1);
		}

		Path outputFilePath = Paths.get(args[6]);
		if (Files.exists(outputFilePath)) {
			System.err.println("Output file already exist on path: " + outputFilePath.toString());
			System.exit(1);
		}

		double[][] data = readIrisDataFile(dataFilePath);

		final KmeansAlgo kmeans = new KmeansAlgo();
		if ("random".equals(initMethod))
			kmeans.initRandomCluster(data);
		else
			kmeans.initPartition(data);

		boolean done = false;
		int iterationCount = 1;

		while (iterationCount <= maxKmeanIteration && !done) {

			done = kmeans.assignmentStep();
			if (!done) {
				done = kmeans.updateStep();
			}
			iterationCount++;
		}

		writeOutputFile(outputFilePath);
	}

	private static double[][] readIrisDataFile(Path dataFilePath) {

		try (BufferedReader reader = Files.newBufferedReader(dataFilePath, Charset.defaultCharset())) {
			double[][] data = new double[150][4];
			String line;
			int datapointCount = 0;
			while ((line = reader.readLine()) != null) {
				String[] features = line.split("\\t");

				for (int featureCount = 0; featureCount < 4; featureCount++) {
					data[datapointCount][featureCount] = Double.parseDouble(features[featureCount]);
				}
				datapointCount++;
			}
			return data;
		} catch (IOException e) {
			e.printStackTrace();
		}
		return null;
	}

	private static void writeOutputFile(Path outputFilePath) {

		try (BufferedWriter writer = Files.newBufferedWriter(outputFilePath, StandardCharsets.UTF_8,
				StandardOpenOption.CREATE_NEW)) {
			writer.write(String.format("k = %s%ndistance = %s%n", k, distance));
			int clusterNo = 1;
			for (final ClusterModel cluster : clusters) {
				writer.write(String.format("centroid %d = %s%n", clusterNo++, Arrays.toString(cluster.getCentroid())));
			}
			writer.write(String.format("mean intercluster distance = %s%n", calMeanInterClusterDist()));
			writer.write(String.format("mean intracluster distance = %s", calMeanIntraClusterDist()));
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public void initRandom(double[][] data) {

		clusters = new ClusterModel[k];

		final Set<Integer> usedObservations = new HashSet<Integer>();

		for (int i = 0; i < k; i++) {
			final ClusterModel cluster = new ClusterModel();

			clusters[i] = cluster;

			int randomDataIndex = -1;
			while (randomDataIndex == -1) {
				randomDataIndex = this.randomInt(0, 149);
				if (usedObservations.contains(randomDataIndex)) {
					randomDataIndex = -1;
				}
			}
			System.arraycopy(data[randomDataIndex], 0, cluster.getCentroid(), 0, data[randomDataIndex].length);
			usedObservations.add(randomDataIndex);
		}

		for (final double[] datapoint : data) {
			final ClusterModel cluster = findNearestCluster(datapoint);
			cluster.getClusterData().add(datapoint);
		}

		updateStep();
	}

	public void initPartition(double[][] data) {

		clusters = new ClusterModel[k];

		int dimension = data[0].length;
		double[] minDimensions = new double[dimension];
		double[] maxDimensions = new double[dimension];
		double[] partitionWidths = new double[dimension];
		for (int i = 0; i < dimension; i++) {
			minDimensions[i] = findMinFromDimension(data, i);
			maxDimensions[i] = findMaxFromDimension(data, i);
			partitionWidths[i] = (maxDimensions[i] - minDimensions[i]) / k;
		}
		double[][] boundryPoints = new double[k + 1][dimension];
		for (int i = 0; i < k + 1; i++) {
			for (int j = 0; j < dimension; j++)
				boundryPoints[i][j] = minDimensions[j] + i * partitionWidths[j];
		}

		for (int i = 0; i < k; i++) {
			final ClusterModel cluster = new ClusterModel();
			clusters[i] = cluster;
			double[] centroid = new double[dimension];
			for (int j = 0; j < dimension; j++) {
				centroid[j] = (boundryPoints[i][j] + boundryPoints[i + 1][j]) / 2.0;
			}
			System.arraycopy(centroid, 0, clusters[i].getCentroid(), 0, dimension);
		}

		for (final double[] datapoint : data) {
			final ClusterModel cluster = findNearestCluster(datapoint);
			cluster.getClusterData().add(datapoint);
		}

		updateStep();
	}

	private boolean updateStep() {
		boolean[] doneClusterWise = new boolean[k];
		int i = 0;
		for (final ClusterModel cluster : clusters) {
			doneClusterWise[i++] = cluster.calculateCentroid(delta);
		}

		for (boolean done : doneClusterWise)
			if (done)
				return true;
		return false;
	}

	private boolean assignmentStep() {
		boolean done = true;

		for (final ClusterModel cluster : clusters) {
			int datapointIndex = 0;
			int datapointCount = cluster.getClusterData().size();

			if (datapointCount > 1) {
				while (datapointIndex < datapointCount) {
					final double[] datapoint = cluster.getClusterData().get(datapointIndex++);

					final ClusterModel targetCluster = findNearestCluster(datapoint);
					if (targetCluster != cluster) {
						cluster.removeClusterDatapoint(datapoint);
						targetCluster.getClusterData().add(datapoint);
						datapointCount--;
						done = false;
					}
				}
			}
		}
		return done;
	}

	private ClusterModel findNearestCluster(final double[] datapoint) {
		ClusterModel result = null;
		double resultDist = Double.POSITIVE_INFINITY;

		for (final ClusterModel cluster : clusters) {
			final double dist = calculateDistance(datapoint, cluster.getCentroid());
			if (dist < resultDist) {
				resultDist = dist;
				result = cluster;
			}
		}

		return result;
	}

	public void initRandomCluster(double[][] data) {

		clusters = new ClusterModel[k];

		for (int i = 0; i < k; i++) {

			clusters[i] = new ClusterModel();

		}

		// assign each datapoint to a random cluster

		for (final double[] datapoint : data) {

			clusters[randomInt(0, k - 1)].getClusterData().add(datapoint);

		}

		// handle any empty clusters

		handleEmptyClusters();

		// calculate initial centers

		updateStep();

	}

	private void handleEmptyClusters() {

		// handle any empty clusters

		for (final ClusterModel cluster : this.clusters) {

			if (cluster.getClusterData().size() == 0) {

				boolean done = false;

				while (!done) {

					final ClusterModel source = clusters[randomInt(0, k - 1)];

					if (source != cluster && source.getClusterData().size() > 1) {

						double[] datapoint = source.getClusterData()
								.get(randomInt(0, source.getClusterData().size() - 1));

						source.removeClusterDatapoint(datapoint);

						cluster.getClusterData().add(datapoint);

						done = true;

					}

				}

			}

		}

	}

	private double findMinFromDimension(double[][] data, int dimension) {
		double min = Double.MAX_VALUE;
		for (double[] datapoints : data) {
			if (Double.compare(datapoints[dimension], min) < 0) {
				min = datapoints[dimension];
			}
		}
		return min;
	}

	private double findMaxFromDimension(double[][] data, int dimension) {
		double max = Double.MIN_VALUE;
		for (double[] datapoints : data) {
			if (Double.compare(datapoints[dimension], max) > 0) {
				max = datapoints[dimension];
			}
		}
		return max;
	}

	public static double calMeanInterClusterDist() {
		double totalDist = 0.0;
		for (int i = 0; i < k; i++) {
			for (int j = i + 1; j < k; j++) {

				totalDist += calEuclideanDistance(clusters[i].getCentroid(), clusters[j].getCentroid());
			}
		}
		return (totalDist / (k * (k - 1) / 2.0));
	}

	public static double calMeanIntraClusterDist() {
		double totalIntraClusterDist = 0.0;
		for (final ClusterModel cluster : clusters) {
			double totalDist = 0.0;
			for (double[] datapoint : cluster.getClusterData()) {
				totalDist += calEuclideanDistance(datapoint, cluster.getCentroid());
			}
			cluster.setMeanIntraClusterDist(totalDist / cluster.getClusterData().size());
			totalIntraClusterDist += cluster.getMeanIntraClusterDist();
		}
		return totalIntraClusterDist / k;
	}

	private static double calculateDistance(double[] vectorA, double[] vectorB) {
		if ("euclidean".equals(distance))
			return calEuclideanDistance(vectorA, vectorB);
		else
			return calCosineDistance(vectorA, vectorB);
	}

	private static double calCosineDistance(double[] vectorA, double[] vectorB) {
		double dotProduct = 0.0;
		double normA = 0.0;
		double normB = 0.0;
		for (int i = 0; i < vectorA.length - 1; i++) {
			dotProduct += vectorA[i] * vectorB[i];
			normA += Math.pow(vectorA[i], 2);
			normB += Math.pow(vectorB[i], 2);
		}
		return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
	}

	private static double calEuclideanDistance(double[] vectorA, double[] vectorB) {
		double total = 0.0;
		for (int i = 0; i < vectorA.length - 1; i++) {
			total += Math.pow(vectorA[i] - vectorB[i], 2);
		}
		return Math.sqrt(total);
	}

	public static int randomInt(int min, int max) {

		Random rand = new Random();
		int randomNum = rand.nextInt((max - min) + 1) + min;

		return randomNum;
	}

	private static void printMatrix(double[][] matrix) {
		System.out.println("-------------------------");
		int i = 1;
		for (double[] datapoints : matrix) {
			System.out.print((i++) + "-> ");
			for (double feature : datapoints)
				System.out.print(feature + " ");
			System.out.println();
		}
		System.out.println("-------------------------");
	}

	private double[][] copyMatrix(double[][] src, int start, int end) {
		double[][] target = new double[(end - start) + 1][src[0].length];
		for (int i = start; i <= end; i++) {
			System.arraycopy(src[i], 0, target[i - start], 0, src[i].length);
		}
		return target;
	}

	private double[][] appendMatrix(double[][] a, double[][] b) {
		double[][] result = new double[a.length + b.length][];
		System.arraycopy(a, 0, result, 0, a.length);
		System.arraycopy(b, 0, result, a.length, b.length);
		return result;
	}

}

/*
 * Cluster Model getters and setters
 * 
 */
class ClusterModel {

	double[] centroid;
	List<double[]> clusterData;
	double meanIntraClusterDist;

	public ClusterModel() {
		this.centroid = new double[4];
		this.clusterData = new ArrayList<>();
		this.meanIntraClusterDist = 0.0;
	}

	public double[] getCentroid() {
		return centroid;
	}

	public void setCentroid(double[] centroid) {
		this.centroid = centroid;
	}

	public List<double[]> getClusterData() {
		return clusterData;
	}

	public void setClusterData(List<double[]> clusterData) {
		this.clusterData = clusterData;
	}

	public double getMeanIntraClusterDist() {
		return meanIntraClusterDist;
	}

	public void setMeanIntraClusterDist(double meanIntraClusterDist) {
		this.meanIntraClusterDist = meanIntraClusterDist;
	}

	public boolean calculateCentroid(double delta) {

		double[] newCentroid = new double[centroid.length];
		int lessThanDeltaCount = 0;
		// First, reset the centroid to zero.
		for (int i = 0; i < centroid.length; i++) {
			this.centroid[i] = 0;
		}
		for (double[] datapoint : this.clusterData) {
			for (int i = 0; i < centroid.length; i++) {
				newCentroid[i] += datapoint[i];
			}
		}
		for (int i = 0; i < centroid.length; i++) {
			newCentroid[i] /= this.clusterData.size();

			if (Math.abs(newCentroid[i] - this.centroid[i]) <= delta)
				lessThanDeltaCount++;
		}

		System.arraycopy(newCentroid, 0, this.centroid, 0, this.centroid.length);

		return (this.centroid.length == lessThanDeltaCount);
	}

	public void removeClusterDatapoint(double[] datapoint) {
		int clusterDataIndex = 0;
		boolean foundDatapoint = false;
		for (; clusterDataIndex < this.clusterData.size(); clusterDataIndex++) {
			if (Arrays.equals(this.clusterData.get(clusterDataIndex), datapoint)) {
				foundDatapoint = true;
				break;
			}
		}

		if (foundDatapoint)
			this.clusterData.remove(clusterDataIndex);
	}

	@Override
	public String toString() {
		final StringBuilder result = new StringBuilder();
		result.append("[Cluster: dimensions=");
		result.append(centroid.length);
		result.append(", observations=");
		result.append(this.clusterData.size());
		result.append(", centroid=");
		result.append(Arrays.toString(this.centroid));
		result.append("]");
		return result.toString();
	}
}

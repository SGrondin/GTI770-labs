import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.Vote;
import weka.core.Instances;

public class PredictMusic {
	public static void main(String[] args) throws Exception {
		if (args.length < 6) {
			System.out.println("Usage : [dossier des données d'entraînement] [dossier des données de test] [dossier où les modèles sont sauvegardés] [fichier de résultats] [faire apprentissage] [faire évaluation]");
			return;
		}
		
		boolean build = args[4].toLowerCase().equals("true");
		boolean evaluate = args[5].toLowerCase().equals("true");
		
		String inBuildFolder = args[0];
		String modelFolder = args[1];
		String testFolder = args[2];
		
		DataModel allModel = new DataModel(inBuildFolder);
		DataModel testModel = new DataModel(testFolder);
		
		DataModel buildModel = allModel;
		DataModel evalModel = testModel;
		
		InstanceUtils.standardize(buildModel, evalModel);
		
		FileOutputStream outStream = new FileOutputStream(new File(args[3]));
		
		if (build)
			buildModel(buildModel, modelFolder);
		
		if (evaluate)
			evaluateModel(buildModel, evalModel, modelFolder, outStream);
	}
	
	private static void evaluateModel(DataModel buildModel, DataModel evalModel, String modelFolder, FileOutputStream outStream) throws Exception {
		Instances combined = InstanceUtils.mergeInstances(evalModel.jmirmfccs, evalModel.marsyas, evalModel.ssd, evalModel.rh);
		ObjectInputStream ois = new ObjectInputStream(new FileInputStream(modelFolder + "all.model"));
		Classifier classifier = (Classifier) ois.readObject();
		ois.close();
		
		Evaluation evaluation = new Evaluation(combined);
		double[] result = evaluation.evaluateModel(classifier, combined);
		System.out.println(evaluation.toSummaryString());
		System.out.println(evaluation.toMatrixString());
		
		System.out.println("Writing results to file ...");
		
		for (int i=0; i<result.length; i++) {
			String res = combined.classAttribute().value((int)result[i]) + "\n";
			outStream.write(res.getBytes());
		}
	}
	
	private static void writeModel(String modelFolder, String name, Classifier data) throws IOException {
		ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(modelFolder + name));
		oos.writeObject(data);
		oos.flush();
		oos.close();
	}
	
	private static void buildModel(DataModel buildModel, String modelFolder) throws Exception {
		ObjectOutputStream oos;
		
		System.out.println("Building Bayes Model ...");
		
		// Modele pour Bayes
		Classifier bayes = Strategy.strategyBayes(buildModel.jmirmfccs, buildModel.marsyas, buildModel.ssd);
		writeModel(modelFolder, "bayes.model", bayes);
		
		System.out.println("Building KNN Model ...");
		
		// Modele pour KNN
		Instances standard_jmirmfccs = InstanceUtils.standardize(buildModel.jmirmfccs, buildModel.jmirmfccs);
		Instances standard_marsyas = InstanceUtils.standardize(buildModel.marsyas, buildModel.marsyas);
		Instances standard_ssd = InstanceUtils.standardize(buildModel.ssd, buildModel.ssd);
		
		Classifier knn = Strategy.strategyKNN(standard_jmirmfccs, standard_marsyas, standard_ssd);
		writeModel(modelFolder, "knn.model", knn);
		
		System.out.println("Building SVM Model ...");
		
		// Modele pour SVM
		Classifier svm = Strategy.strategySVM(buildModel.rh);
		writeModel(modelFolder, "svm.model", svm);
		
		// Model qui utilise les trois strategies
		Vote combinedModel = new Vote();

		combinedModel.setClassifiers(new Classifier[] { bayes, knn, svm });
		writeModel(modelFolder, "all.model", combinedModel);
	}
}

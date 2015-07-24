import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

public class PredictMusic {
	public static void main(String[] args) throws Exception {
		boolean build = true;
		boolean evaluate = true;
		
		String inBuildFolder = "../Sample/";
		String inEvalFolder = "../Sample/";
		String modelFolder = "../Model/";
		
		DataModel buildModel = new DataModel(inBuildFolder);
		DataModel evalModel = new DataModel(inEvalFolder);

		if (build)
			buildModel(buildModel, modelFolder);
		
		if (evaluate)
			evaluateModel(buildModel, evalModel, modelFolder);
	}
	
	private static void evaluateModel(DataModel buildModel, DataModel evalModel, String modelFolder) throws Exception {
		Instances combined;
		ObjectInputStream ois;
		
		// Les trois premiers modeles sont utilisent ensemble pour 2 classificateurs
		combined = InstanceUtils.mergeInstances(evalModel.jmirmfccs, evalModel.marsyas, evalModel.ssd);
		
		// Bayes !
		ois = new ObjectInputStream(new FileInputStream(modelFolder + "bayes.model"));
		Classifier classifier = (Classifier) ois.readObject();
		
		Evaluation evaluation = new Evaluation(combined);
		evaluation.evaluateModel(classifier, combined);
		System.out.println(evaluation.toSummaryString());
		
		// KNN !
		Instances s_jmirmfccs = InstanceUtils.standardize(buildModel.jmirmfccs, evalModel.jmirmfccs);
		Instances s_marsyas = InstanceUtils.standardize(buildModel.marsyas, evalModel.marsyas);
		Instances s_ssd = InstanceUtils.standardize(buildModel.ssd, evalModel.ssd);
		combined = InstanceUtils.mergeInstances(s_jmirmfccs, s_marsyas, s_ssd);
		
		ois = new ObjectInputStream(new FileInputStream(modelFolder + "knn.model"));
		classifier = (Classifier) ois.readObject();
		
		evaluation = new Evaluation(combined);
		evaluation.evaluateModel(classifier, combined);
		System.out.println(evaluation.toSummaryString());
	}
	
	private static void buildModel(DataModel buildModel, String modelFolder) throws Exception {
		ObjectOutputStream oos;
		
		// Modele pour Bayes
		Classifier bayes = Strategy.strategyBayes(buildModel.jmirmfccs, buildModel.marsyas, buildModel.ssd);
		oos = new ObjectOutputStream(new FileOutputStream(modelFolder + "bayes.model"));
		oos.writeObject(bayes);
		oos.flush();
		oos.close();
		
		// Modele pour KNN
		Instances standard_jmirmfccs = InstanceUtils.standardize(buildModel.jmirmfccs, buildModel.jmirmfccs);
		Instances standard_marsyas = InstanceUtils.standardize(buildModel.marsyas, buildModel.marsyas);
		Instances standard_ssd = InstanceUtils.standardize(buildModel.ssd, buildModel.ssd);
		
		Classifier knn = Strategy.strategyKNN(standard_jmirmfccs, standard_marsyas, standard_ssd);
		oos = new ObjectOutputStream(new FileOutputStream(modelFolder + "knn.model"));
		oos.writeObject(knn);
		oos.flush();
		oos.close();
	}
}

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.Vote;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.Standardize;


public class PredictMusic {
	public static void main(String[] args) throws Exception {
		String step = "evaluate";
		String inBuildFolder = "../Sample/";
		String inEvalFolder = "../Sample/";
		String modelFolder = "../Model/";
		ObjectOutputStream oos;
		ObjectInputStream ois;
		Instances combined;
		
		String folder = "build".equals(step) ? inBuildFolder : inEvalFolder;
		
		Instances jmirmfccs = DataSource.read(folder + "msd-jmirmfccs_dev.arff");
		Instances marsyas = DataSource.read(folder + "msd-marsyas_dev_new.arff");
		Instances ssd = DataSource.read(folder + "msd-ssd_dev.arff");
		Instances rh = DataSource.read(folder + "msd-rh_dev_new.arff");
		
		// On enlève les attributes textes qui ne sont pas pertinents //
		jmirmfccs = InstanceUtils.removeIdentifier(jmirmfccs);
		marsyas = InstanceUtils.removeIdentifier(marsyas);
		ssd = InstanceUtils.removeIdentifier(ssd);
		rh = InstanceUtils.removeIdentifier(rh);
		
		if ("build".equals(step)) {
			// Modele pour Bayes
			Classifier bayes = Strategy.strategyBayes(jmirmfccs, marsyas, ssd);
			oos = new ObjectOutputStream(new FileOutputStream(modelFolder + "bayes.model"));
			oos.writeObject(bayes);
			oos.flush();
			oos.close();
			
			// Modele pour KNN
			Instances s_jmirmfccs = InstanceUtils.standardize(jmirmfccs, jmirmfccs);
			Instances s_marsyas = InstanceUtils.standardize(marsyas, marsyas);
			Instances s_ssd = InstanceUtils.standardize(ssd, ssd);
			
			Classifier knn = Strategy.strategyKNN(s_jmirmfccs, s_marsyas, s_ssd);
			oos = new ObjectOutputStream(new FileOutputStream(modelFolder + "knn.model"));
			oos.writeObject(knn);
			oos.flush();
			oos.close();
		}
		
		if ("evaluate".equals(step)) {
			// Les trois premiers modeles sont utilisent ensemble pour 2 classificateurs
			combined = InstanceUtils.mergeInstances(jmirmfccs, marsyas, ssd);
			
			// Bayes !
			ois = new ObjectInputStream(new FileInputStream(modelFolder + "bayes.model"));
			Classifier classifier = (Classifier) ois.readObject();
			
			Evaluation evaluation = new Evaluation(combined);
			evaluation.evaluateModel(classifier, combined);
			System.out.println(evaluation.toSummaryString());
			
			// KNN !
			Instances s_jmirmfccs = InstanceUtils.standardize(jmirmfccs, jmirmfccs);
			Instances s_marsyas = InstanceUtils.standardize(marsyas, marsyas);
			Instances s_ssd = InstanceUtils.standardize(ssd, ssd);
			combined = InstanceUtils.mergeInstances(s_jmirmfccs, s_marsyas, s_ssd);
			
			ois = new ObjectInputStream(new FileInputStream(modelFolder + "knn.model"));
			classifier = (Classifier) ois.readObject();
			
			evaluation = new Evaluation(combined);
			evaluation.evaluateModel(classifier, combined);
			System.out.println(evaluation.toSummaryString());
		}
		
	}
}

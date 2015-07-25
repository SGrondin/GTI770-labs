import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.Vote;
import weka.core.Instances;


public class Strategy {
	public static Classifier strategyKNN(Instances... set) throws Exception {
		Classifier[] classifiers = new Classifier[set.length];
		
		for (int i=0; i<set.length; i++) {
			Instances inst = set[i];
			IBk ibk = new IBk();
			ibk.setOptions(new String[]{ "-K", "3", "-W", "0", "-A", "weka.core.neighboursearch.LinearNNSearch -A \"weka.core.EuclideanDistance -R first-last\"" });
			
			Classifier classifier = new AttributeClassifier(ibk);
			classifier.buildClassifier(inst);
			classifiers[i] = classifier;
		}
		
		Vote vote = new Vote();
		vote.setClassifiers(classifiers);
		return vote;
	}
	
	public static Classifier strategyBayes(Instances... set) throws Exception {
		Classifier[] classifiers = new Classifier[set.length];
		
		for (int i=0; i<set.length; i++) {
			Instances inst = set[i];
			NaiveBayes bayes = new NaiveBayes();
			bayes.setOptions(new String[]{ "-D" });
			
			Classifier classifier = new AttributeClassifier(bayes);
			classifier.buildClassifier(inst);
			classifiers[i] = classifier;
		}
		
		Vote vote = new Vote();
		vote.setClassifiers(classifiers);
		return vote;
	}
	
	public static Classifier strategySVM(Instances... set) throws Exception {
		Instances inst = InstanceUtils.mergeInstances(set);
		
		LibSVM svm = new LibSVM();
		svm.setOptions(new String[]{ "-S", "0", "-K", "2", "-D", "3", "-G", "0.0", "-R", "0.0", "-N", "0.5", "-M", "40.0", "-C", "1.0", "-E", "0.001", "-P", "0.1", "-seed", "1" });
		
		Classifier classifier = new AttributeClassifier(svm);
		classifier.buildClassifier(inst);
		return classifier;
	}
}

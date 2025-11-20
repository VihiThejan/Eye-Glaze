import { useEffect, useState } from 'react';
import { format } from 'date-fns';
import { Eye, Brain, Activity } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { toast } from 'sonner';
import { User } from '@/types';

interface LatestAnalysis {
  _id: string;
  username: string;
  hasStress: boolean;
  imageUrl: string | null;
  createdAt: string;
}

interface LatestEyeImage {
  _id: string;
  username: string;
  imageUrl: string;
  cloudinaryId: string;
  uploadedAt: string;
}

interface AnalysisCount {
  username: string;
  totalAnalyses: number;
}

export function AnalysisResults() {
  const [user, setUser] = useState<User | null>(null);
  const [latestAnalysis, setLatestAnalysis] = useState<LatestAnalysis | null>(null);
  const [latestImage, setLatestImage] = useState<LatestEyeImage | null>(null);
  const [analysisCount, setAnalysisCount] = useState<AnalysisCount | null>(null);
  const [loading, setLoading] = useState(true);

  // Mock data for detailed analysis (would come from API in real implementation)
  const detailedAnalysis = {
    stressLevel: 3.2,
    metrics: [
      { 
        label: 'Pupil Dilation', 
        value: 6.5, 
        icon: Eye,
        description: 'Measures autonomic nervous system response'
      },
      { 
        label: 'Eye Movement', 
        value: 4.2, 
        icon: Activity,
        description: 'Tracks micro-movements and fixation patterns'
      },
      { 
        label: 'Blink Rate', 
        value: 7.8, 
        icon: Brain,
        description: 'Indicates cognitive load and stress levels'
      },
    ]
  };

  useEffect(() => {
    // Get user from localStorage
    const getUserFromStorage = (): User | null => {
      const storedUser = localStorage.getItem('eyeGlazeUser');
      return storedUser ? JSON.parse(storedUser) : null;
    };
    
    setUser(getUserFromStorage());
  }, []);

  const fetchLatestAnalysis = async (username: string) => {
    try {
      const response = await fetch(`http://localhost:5176/api/analysis/latest/${username}`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch latest analysis: ${response.status}`);
      }
      
      const result = await response.json();
      if (result.status === "success" && result.data) {
        setLatestAnalysis(result.data);
      }
    } catch (error) {
      console.error("Error fetching latest analysis:", error);
      toast.error("Could not retrieve latest analysis data");
    }
  };

  const fetchLatestImage = async (username: string) => {
    try {
      const response = await fetch(`http://localhost:5176/api/upload/eye-image/latest/${username}`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch latest eye image: ${response.status}`);
      }
      
      const result = await response.json();
      if (result.status === "success" && result.data) {
        setLatestImage(result.data);
      }
    } catch (error) {
      console.error("Error fetching latest eye image:", error);
      // Don't show toast here as image might not always be available
    }
  };

  const fetchAnalysisCount = async (username: string) => {
    try {
      const response = await fetch(`http://localhost:5176/api/analysis/count/${username}`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch analysis count: ${response.status}`);
      }
      
      const result = await response.json();
      if (result.status === "success" && result.data) {
        setAnalysisCount(result.data);
      }
    } catch (error) {
      console.error("Error fetching analysis count:", error);
      toast.error("Could not retrieve analysis count");
    }
  };

  useEffect(() => {
    const loadData = async () => {
      if (!user?.email) return;

      setLoading(true);
      
      try {
        await Promise.all([
          fetchLatestAnalysis(user.email),
          fetchLatestImage(user.email),
          fetchAnalysisCount(user.email)
        ]);
      } finally {
        setLoading(false);
      }
    };

    if (user) {
      loadData();
    }
  }, [user]);

  const getStressColor = (hasStress: boolean) => {
    return hasStress ? 'text-red-400' : 'text-green-400';
  };

  const getStressLevel = (hasStress: boolean) => {
    return hasStress ? 'Moderate' : 'Low';
  };

  const getStressBadgeVariant = (hasStress: boolean) => {
    return hasStress ? 'secondary' : 'default';
  };

  // Generate analysis ID from database ID or create a placeholder
  const getAnalysisId = () => {
    if (!latestAnalysis?._id) return "#EG-0000-000";
    
    // Extract last 6 chars from MongoDB ID and format as EG-YYYY-XXX
    const year = new Date().getFullYear();
    const idSuffix = latestAnalysis._id.slice(-6);
    return `#EG-${year}-${idSuffix}`;
  };

  if (loading) {
    return (
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2 text-2xl">
            <Eye className="h-6 w-6" />
            <span>Loading Analysis...</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center p-12">
            <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary"></div>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!latestAnalysis) {
    return (
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2 text-2xl">
            <Eye className="h-6 w-6" />
            <span>No Analysis Found</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground">
            You haven't completed any eye analyses yet. Complete your first scan to see results here.
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="glass-card">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center space-x-2 text-2xl">
              <Eye className="h-6 w-6" />
              <span>Latest Analysis</span>
            </CardTitle>
            <CardDescription className="text-base">
              {format(new Date(latestAnalysis.createdAt), "MMM d, yyyy 'at' h:mm a")} • Analysis ID: {getAnalysisId()}
            </CardDescription>
          </div>
          <Badge variant={getStressBadgeVariant(latestAnalysis.hasStress)} className="px-4 py-2 text-sm">
            {getStressLevel(latestAnalysis.hasStress)} Stress
          </Badge>
        </div>
      </CardHeader>
      <CardContent>
        <div className="grid lg:grid-cols-2 gap-12">
          {/* Stress Level Display with Image */}
          <div className="space-y-8">
            <div className="text-center">
              <div className={`text-8xl font-bold ${getStressColor(latestAnalysis.hasStress)} mb-4`}>
                {latestAnalysis.hasStress ? '3.2' : '2.1'}
              </div>
              <div className="text-xl text-muted-foreground mb-6">
                Stress Level (out of 10)
              </div>
              <Progress 
                value={latestAnalysis.hasStress ? 32 : 21} 
                className="w-full h-4 mb-4"
              />
              <p className="text-sm text-muted-foreground">
                Your stress level is currently in the {getStressLevel(latestAnalysis.hasStress).toLowerCase()} range
              </p>
            </div>

            {/* Display eye image if available */}
            {(latestAnalysis.imageUrl || latestImage?.imageUrl) && (
              <div className="mt-6">
                <h3 className="text-lg font-medium mb-3">Eye Scan Image</h3>
                <div className="relative rounded-lg overflow-hidden" style={{ height: "200px" }}>
                  <img 
                    src={latestAnalysis.imageUrl || latestImage?.imageUrl} 
                    alt="Eye scan" 
                    className="absolute inset-0 w-full h-full object-cover"
                  />
                </div>
                <p className="text-xs text-muted-foreground mt-2">
                  Analysis based on eye scan taken {format(new Date(latestAnalysis.createdAt), "MMMM d, yyyy 'at' h:mm a")}
                </p>
              </div>
            )}
          </div>

          {/* Analysis Metrics */}
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h3 className="text-2xl font-semibold">Detailed Analysis</h3>
              {analysisCount && (
                <Badge variant="outline" className="px-3 py-1">
                  {analysisCount.totalAnalyses} total {analysisCount.totalAnalyses === 1 ? 'analysis' : 'analyses'}
                </Badge>
              )}
            </div>
            
            <div className="space-y-6">
              {detailedAnalysis.metrics.map((metric, index) => (
                <div key={index} className="space-y-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <div className="w-10 h-10 rounded-xl bg-primary/20 flex items-center justify-center">
                        <metric.icon className="h-5 w-5 text-primary" />
                      </div>
                      <div>
                        <span className="font-medium">{metric.label}</span>
                        <p className="text-xs text-muted-foreground">{metric.description}</p>
                      </div>
                    </div>
                    <div className="text-right">
                      <span className="text-lg font-semibold">{metric.value}</span>
                      <div className="text-xs text-muted-foreground">out of 10</div>
                    </div>
                  </div>
                  <Progress value={metric.value * 10} className="w-full h-2" />
                </div>
              ))}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
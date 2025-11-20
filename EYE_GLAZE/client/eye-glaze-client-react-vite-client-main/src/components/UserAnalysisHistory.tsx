import { useState, useEffect } from 'react';
import { useAuth } from '@/contexts/AuthContext';
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Calendar, Clock, AlertTriangle, Check, AlertCircle, ArrowUp, ArrowDown, Eye } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { toast } from 'sonner';

interface AnalysisItem {
  _id: string;
  username: string;
  hasStress: boolean;
  imageUrl: string | null;
  confidenceLevel?: number;
  createdAt: string;
}

export function UserAnalysisHistory() {
  const { user } = useAuth();
  const [analyses, setAnalyses] = useState<AnalysisItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'grid' | 'timeline'>('grid');

  useEffect(() => {
    const fetchAnalyses = async () => {
      if (!user?.email) {
        setLoading(false);
        return;
      }

      try {
        const response = await fetch(`http://localhost:5176/api/analysis/user/${encodeURIComponent(user.email)}`);
        
        if (!response.ok) {
          throw new Error('Failed to fetch analysis history');
        }
        
        const data = await response.json();
        
        if (data.status === 'success') {
          // Sort analyses by date (newest first)
          const sortedAnalyses = data.data.sort((a: AnalysisItem, b: AnalysisItem) => 
            new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
          );
          setAnalyses(sortedAnalyses);
        } else {
          throw new Error(data.message || 'Failed to fetch analysis history');
        }
      } catch (error) {
        console.error('Error fetching analyses:', error);
        toast.error('Could not load your analysis history. Please try again later.');
      } finally {
        setLoading(false);
      }
    };

    fetchAnalyses();
  }, [user?.email]);

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return new Intl.DateTimeFormat('en-US', { 
      dateStyle: 'medium',
      timeStyle: 'short'
    }).format(date);
  };

  const getTimeAgo = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffSecs = Math.floor(diffMs / 1000);
    const diffMins = Math.floor(diffSecs / 60);
    const diffHours = Math.floor(diffMins / 60);
    const diffDays = Math.floor(diffHours / 24);

    if (diffDays > 0) {
      return `${diffDays} day${diffDays > 1 ? 's' : ''} ago`;
    } else if (diffHours > 0) {
      return `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
    } else if (diffMins > 0) {
      return `${diffMins} minute${diffMins > 1 ? 's' : ''} ago`;
    } else {
      return 'Just now';
    }
  };

  // Group analyses by date
  const analysesByDate = analyses.reduce((acc, analysis) => {
    const date = new Date(analysis.createdAt).toLocaleDateString();
    if (!acc[date]) {
      acc[date] = [];
    }
    acc[date].push(analysis);
    return acc;
  }, {} as Record<string, AnalysisItem[]>);

  // Calculate stress trends
  const stressedCount = analyses.filter(a => a.hasStress).length;
  const stressPercentage = analyses.length > 0 ? Math.round((stressedCount / analyses.length) * 100) : 0;
  
  // Determine if stress is increasing or decreasing based on the last 5 analyses
  let trendDirection: 'up' | 'down' | 'stable' = 'stable';
  if (analyses.length >= 5) {
    const recentFive = analyses.slice(0, 5);
    const olderFive = analyses.slice(Math.max(0, analyses.length - 5));
    
    const recentStressCount = recentFive.filter(a => a.hasStress).length;
    const olderStressCount = olderFive.filter(a => a.hasStress).length;
    
    if (recentStressCount > olderStressCount) {
      trendDirection = 'up';
    } else if (recentStressCount < olderStressCount) {
      trendDirection = 'down';
    }
  }

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-pulse text-primary font-medium">Loading your analysis history...</div>
      </div>
    );
  }

  if (analyses.length === 0) {
    return (
      <Alert className="bg-primary/5 border border-primary/20">
        <AlertCircle className="h-5 w-5 text-primary" />
        <AlertDescription>
          You haven't submitted any eye analyses yet. Try uploading an eye image in the "Upload & Analyze" tab to get started.
        </AlertDescription>
      </Alert>
    );
  }

  return (
    <div className="space-y-8">
      {/* Summary Cards */}
      <div className="grid md:grid-cols-3 gap-4">
        <Card className="bg-primary/5 border border-primary/20">
          <CardHeader className="pb-2">
            <CardTitle className="text-lg">Total Analyses</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{analyses.length}</div>
            <p className="text-sm text-muted-foreground">Eye images analyzed</p>
          </CardContent>
        </Card>
        
        <Card className={`${stressPercentage > 50 ? 'bg-red-500/10 border-red-500/20' : 'bg-green-500/10 border-green-500/20'}`}>
          <CardHeader className="pb-2">
            <CardTitle className="text-lg">Stress Rate</CardTitle>
          </CardHeader>
          <CardContent className="flex items-center gap-2">
            <div className="text-3xl font-bold">{stressPercentage}%</div>
            <div className="flex flex-col">
              <span className="text-sm text-muted-foreground">{stressedCount} of {analyses.length}</span>
              <div className="flex items-center">
                {trendDirection === 'up' && <ArrowUp className="h-3 w-3 text-red-500" />}
                {trendDirection === 'down' && <ArrowDown className="h-3 w-3 text-green-500" />}
                <span className={`text-xs ${trendDirection === 'up' ? 'text-red-500' : trendDirection === 'down' ? 'text-green-500' : 'text-muted-foreground'}`}>
                  {trendDirection === 'up' ? 'Increasing' : trendDirection === 'down' ? 'Decreasing' : 'Stable'}
                </span>
              </div>
            </div>
          </CardContent>
        </Card>
        
        <Card className="bg-background">
          <CardHeader className="pb-2">
            <CardTitle className="text-lg">Last Analysis</CardTitle>
          </CardHeader>
          <CardContent>
            {analyses[0] && (
              <div className="space-y-1">
                <div className={`text-sm font-medium ${analyses[0].hasStress ? 'text-red-500' : 'text-green-500'}`}>
                  {analyses[0].hasStress ? 'Stress Detected' : 'No Stress'}
                </div>
                <div className="text-xs text-muted-foreground flex items-center gap-1">
                  <Clock className="h-3 w-3" /> {getTimeAgo(analyses[0].createdAt)}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* View Mode Selection */}
      <div className="flex justify-end">
        <Tabs value={viewMode} onValueChange={(value) => setViewMode(value as 'grid' | 'timeline')} className="w-[200px]">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="grid">Grid</TabsTrigger>
            <TabsTrigger value="timeline">Timeline</TabsTrigger>
          </TabsList>
        </Tabs>
      </div>

      {/* Image Modal */}
      {selectedImage && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4" onClick={() => setSelectedImage(null)}>
          <div className="relative max-w-3xl max-h-[80vh] overflow-hidden">
            <img src={selectedImage} alt="Eye analysis" className="max-w-full max-h-[80vh] object-contain" />
            <Button 
              variant="outline" 
              className="absolute top-4 right-4 bg-black/50 hover:bg-black/70 text-white border-none"
              onClick={(e) => {
                e.stopPropagation();
                setSelectedImage(null);
              }}
            >
              Close
            </Button>
          </div>
        </div>
      )}

      {/* Grid View */}
      {viewMode === 'grid' && (
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {analyses.map((analysis) => (
            <Card key={analysis._id} className="overflow-hidden hover:shadow-md transition-shadow">
              <div 
                className="w-full h-48 bg-gray-100 dark:bg-gray-800 overflow-hidden relative cursor-pointer"
                onClick={() => analysis.imageUrl && setSelectedImage(analysis.imageUrl)}
              >
                {analysis.imageUrl ? (
                  <img 
                    src={analysis.imageUrl} 
                    alt="Eye analysis" 
                    className="w-full h-full object-cover hover:scale-105 transition-transform" 
                  />
                ) : (
                  <div className="w-full h-full flex items-center justify-center">
                    <Eye className="h-10 w-10 text-muted-foreground/30" />
                  </div>
                )}
                <Badge 
                  variant="outline" 
                  className={`absolute top-2 right-2 ${
                    analysis.hasStress 
                      ? 'bg-red-500/90 text-white border-red-600' 
                      : 'bg-green-500/90 text-white border-green-600'
                  }`}
                >
                  {analysis.hasStress ? 'Stress Detected' : 'No Stress'}
                </Badge>
              </div>
              <CardFooter className="flex justify-between pt-4">
                <div className="text-xs text-muted-foreground flex items-center gap-1">
                  <Calendar className="h-3 w-3" /> {formatDate(analysis.createdAt)}
                </div>
                {analysis.confidenceLevel && (
                  <div className="text-xs font-medium">
                    {Math.round(analysis.confidenceLevel * 100)}% Confidence
                  </div>
                )}
              </CardFooter>
            </Card>
          ))}
        </div>
      )}

      {/* Timeline View */}
      {viewMode === 'timeline' && (
        <div className="space-y-8">
          {Object.entries(analysesByDate).map(([date, items]) => (
            <div key={date} className="space-y-2">
              <h3 className="font-medium text-sm text-muted-foreground">{date}</h3>
              <div className="space-y-4 pl-4 border-l-2 border-primary/20">
                {items.map((analysis) => (
                  <div key={analysis._id} className="relative pl-6">
                    <div className="absolute left-[-9px] top-1/2 transform -translate-y-1/2 w-4 h-4 rounded-full bg-primary"></div>
                    <Card>
                      <CardContent className="p-4 flex items-center gap-4">
                        <div 
                          className="w-16 h-16 bg-gray-100 dark:bg-gray-800 rounded-md overflow-hidden flex-shrink-0 cursor-pointer"
                          onClick={() => analysis.imageUrl && setSelectedImage(analysis.imageUrl)}
                        >
                          {analysis.imageUrl ? (
                            <img 
                              src={analysis.imageUrl} 
                              alt="Eye analysis" 
                              className="w-full h-full object-cover" 
                            />
                          ) : (
                            <div className="w-full h-full flex items-center justify-center">
                              <Eye className="h-5 w-5 text-muted-foreground/30" />
                            </div>
                          )}
                        </div>
                        <div className="flex-grow">
                          <div className="flex justify-between items-start">
                            <div className="flex items-center gap-1">
                              {analysis.hasStress ? (
                                <>
                                  <AlertTriangle className="h-4 w-4 text-red-500" />
                                  <span className="font-medium text-red-500">Stress Detected</span>
                                </>
                              ) : (
                                <>
                                  <Check className="h-4 w-4 text-green-500" />
                                  <span className="font-medium text-green-500">No Stress</span>
                                </>
                              )}
                            </div>
                            <div className="text-xs text-muted-foreground">
                              {new Date(analysis.createdAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                            </div>
                          </div>
                          {analysis.confidenceLevel && (
                            <div className="text-xs mt-1">
                              Analysis confidence: <span className="font-medium">{Math.round(analysis.confidenceLevel * 100)}%</span>
                            </div>
                          )}
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}